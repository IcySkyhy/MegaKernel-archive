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
 * \file intra_subgraph_adapter.cpp
 * \brief
 */

#include <unordered_set>
#include "interface/configs/config_manager_ng.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/tensor/irbuilder.h"
#include "intra_subgraph_adapter.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_check/intra_subgraph_adapter_checker.h"
#include "passes/pass_utils/infer_shape_utils.h"

#define MODULE_NAME "IntraSubgraphAdapter"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
template <typename T>
inline std::string IntSetToStr(const std::set<T>& colorSet)
{
    std::stringstream ss;
    ss << "(";

    if (!colorSet.empty()) {
        for (const auto& x : colorSet) {
            ss << x << ", ";
        }
    }

    ss << ")";
    return ss.str();
}

static bool IsAssembleWithDifferentMemoryType(const Operation& op, MemoryType outputMemoryType)
{
    if (!IsAssembleLike(op.GetOpcode()) || op.GetOpcode() == config::GetContractOpcode() || op.GetIOperands().empty() ||
        op.GetOOperands().empty()) {
        return false;
    }
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    if (input == nullptr || output == nullptr) {
        return false;
    }
    auto inputType = input->GetMemoryTypeOriginal();
    return inputType != MemoryType::MEM_UNKNOWN && outputMemoryType != MemoryType::MEM_UNKNOWN &&
           inputType != outputMemoryType;
}

static bool IsViewWithDifferentMemoryType(const Operation& op, MemoryType inputMemoryType)
{
    if (!IsViewLike(op.GetOpcode()) || op.GetOpcode() == config::GetSliceOpcode() || op.GetIOperands().empty() ||
        op.GetOOperands().empty()) {
        return false;
    }
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    if (input == nullptr || output == nullptr) {
        return false;
    }
    auto outputType = output->GetMemoryTypeOriginal();
    return inputMemoryType != MemoryType::MEM_UNKNOWN && outputType != MemoryType::MEM_UNKNOWN &&
           inputMemoryType != outputType;
}

static void SetConfiguredOpcode(Operation& op, Opcode opcode)
{
    if (op.GetOpcode() == opcode) {
        return;
    }
    std::string oldOpcode = op.GetOpcodeStr();
    op.SetOpCode(opcode);
    APASS_LOG_DEBUG_F(Elements::Operation, "Change %s[%d] opcode to %s.", oldOpcode.c_str(), op.GetOpMagic(),
                      op.GetOpcodeStr().c_str());
}

Status IntraSubgraphAdapter::RunOnFunction(Function& function)
{
    LogicalTensors boundaryTensors = CollectBoundaryTensors(function);

    for (size_t i = 0; i < boundaryTensors.size(); i++) {
        LogicalTensorPtr tensor = boundaryTensors[i];
        if (CheckBoundaryTensor(tensor) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "Check boundary tensor failed; Please check the CheckBoundaryTensor method.");
            return FAILED;
        }

        // If the boundary tensor is already in the ddr, then skip processing this tensor
        if (tensor->GetMemoryTypeOriginal() == MEM_DEVICE_DDR) {
            continue;
        }

        std::set<int> producerColors, consumerColors;
        CollectProducerColors(tensor, producerColors);
        CollectConsumerColors(tensor, consumerColors);

        std::set<int> commonColors = SetIntersection(producerColors, consumerColors);
        if (CheckColorCount(function, tensor, commonColors.size()) != SUCCESS) {
            return FAILED;
        }
        if (commonColors.size() == 1) {
            if (!Platform::Instance().GetDie().HasDirectPath(tensor->GetMemoryTypeOriginal(), MEM_DEVICE_DDR)) {
                APASS_LOG_INFO_F(
                    Elements::Tensor,
                    "No direct path from %s to DDR for boundary tensor %s, process as DDR boundary directly.",
                    MemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str(), tensor->Dump().c_str());
                if (ProcessBoundaryTensor(function, tensor) == FAILED) {
                    APASS_LOG_ERROR_F(
                        Elements::Tensor,
                        "Process boundary tensor failed, tensor magic : %d; Please check ProcessBoundaryTensor.",
                        tensor->GetMagic());
                    return FAILED;
                }
                continue;
            }
            // For boundary tensor that have both producer and consumer in a single subgraph,
            // we split it to multiple boundary tensors, whose producers and consumers do not share same subgraph.
            int mainSubgraphID = *(commonColors.begin()); // the only subgraph id that has both producers and consumers.
            LogicalTensors newBoundaryTensors;

            APASS_LOG_DEBUG_F(Elements::Tensor, "********** %s requires SplitBoundaryTensor. **********",
                              function.GetMagicName().c_str());
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Boundary tensor: %s, mainSubgraphID: %d, producerColors: %s, consumerColors: %s",
                              tensor->Dump().c_str(), mainSubgraphID, IntSetToStr(producerColors).c_str(),
                              IntSetToStr(consumerColors).c_str());
            for (const auto& producer : tensor->GetProducers()) {
                APASS_LOG_DEBUG_F(Elements::Operation, "producer: %s", producer->Dump().c_str());
            }
            for (const auto& consumer : tensor->GetConsumers()) {
                APASS_LOG_DEBUG_F(Elements::Operation, "consumer: %s", consumer->Dump().c_str());
            }

            if (SplitBoundaryTensor(function, tensor, mainSubgraphID, newBoundaryTensors) == FAILED) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "Split boundary tensor failed, tensor magic : %d; Please check SplitBoundaryTensor method.",
                    tensor->GetMagic());
                return FAILED;
            }
            if (ProcessBoundaryTensors(function, newBoundaryTensors) == FAILED) {
                APASS_LOG_ERROR_F(Elements::Tensor,
                                  "Process boundary tensors failed; Please check ProcessBoundaryTensors method.");
                return FAILED;
            }
        }
    }
    if (!newOps.empty()) {
        if (InferShapeUtils::InferShape(function, newOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::CheckColorCount(Function& function, LogicalTensorPtr tensor, size_t commonColorCount)
{
    if (commonColorCount > 1) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "Process boundary tensor failed, tensor magic : %d; The producers and consumers cannot simultaneously "
            "appear in more than one subgraph.",
            tensor->GetMagic());
        return FAILED;
    }
    if (commonColorCount == 0) {
        if (ProcessBoundaryTensor(function, tensor) == FAILED) {
            APASS_LOG_ERROR_F(
                Elements::Tensor,
                "Process boundary tensor failed, tensor magic : %d; Please check the ProcessBoundaryTensor method.",
                tensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}
Status IntraSubgraphAdapter::CheckBoundaryTensor(LogicalTensorPtr tensor)
{
    const static std::unordered_set<MemoryType> validBoundaryTensorMemType = {MEM_UB, MEM_L1, MEM_L0C, MEM_DEVICE_DDR};
    if (tensor->GetMemoryTypeOriginal() != tensor->GetMemoryTypeToBe()) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "There is a conflict in the memory type, tensor magic : %d, tensor original memory type : %s, tensor tobe "
            "memory type : %s; The original memory type and tobe memory type should be consistent.",
            tensor->GetMagic(), MemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str(),
            MemoryTypeToString(tensor->GetMemoryTypeToBe()).c_str());
        return FAILED;
    }
    if (validBoundaryTensorMemType.find(tensor->GetMemoryTypeOriginal()) == validBoundaryTensorMemType.end()) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "Original memory type error, tensor magic : %d, tensor memory type : %s; The original type of boundary "
            "tensor should be in [UB, L1, DDR, L0C].",
            tensor->GetMagic(), MemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::SplitBoundaryTensor(Function& function, LogicalTensorPtr tensor, int mainSubgraphID,
                                                 LogicalTensors& newBoundaryTensors)
{
    // If the tensor has multiple producers in different subgraph, producers must be assemble-family or OP_COPY_OUT.
    for (const auto& producer : tensor->GetProducers()) {
        if (producer->GetSubgraphID() != mainSubgraphID) {
            Opcode producerOpcode = producer->GetOpcode();
            if (OpcodeManager::Inst().GetOpCalcType(producerOpcode) != OpCalcType::MOVE_OUT &&
                OpcodeManager::Inst().GetOpCalcType(producerOpcode) != OpCalcType::MOVE_LOCAL) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "OpCalcType error, boundary tensor magic : %d, producer op magic : %d, producer op : %s; If the "
                    "tensor has multiple producers, then the producers can only be assemble-family/OP_COPY_OUT.%s",
                    tensor->GetMagic(), producer->GetOpMagic(), producer->GetOpcodeStr().c_str(),
                    GetFormatBacktrace(*producer).c_str());
                return FAILED;
            }
            LogicalTensors& producerInputs = producer->GetIOperands();
            if (producerInputs.size() != 1) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Producer input error, boundary tensor magic : %d, producer op magic : %d, producer op : %s; The "
                    "Assemble-family producer should have one input operand.%s",
                    tensor->GetMagic(), producer->GetOpMagic(), producer->GetOpcodeStr().c_str(),
                    GetFormatBacktrace(*producer).c_str());
                return FAILED;
            }
            // When there are multiple consumers, for producer from other subgraph, we insert a new ASSEMBLE
            // to the other subgraph, and change the producer to main subgraph.
            if (tensor->GetConsumers().size() != 1) {
                LogicalTensorPtr assembleInput = producer->GetIOperands()[0];
                APASS_LOG_DEBUG_F(Elements::Tensor, "SplitBoundaryTensor output of %s[%d]",
                                  producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
                LogicalTensorPtr newTensor = InsertOpBetween(function, config::GetContractOpcode(), assembleInput,
                                                             {producer});
                producer->UpdateSubgraphID(mainSubgraphID);
                APASS_LOG_INFO_F(Elements::Tensor, "Adjust %s(magic : %d) to subgraph %d.",
                                 producer->GetOpcodeStr().c_str(), producer->GetOpMagic(), mainSubgraphID);
                // The intermediate tensor become a new boundary tensor.
                newBoundaryTensors.push_back(newTensor);
                APASS_LOG_INFO_F(Elements::Tensor, "Add new tensor(magic : %d) to boundary tensors.",
                                 assembleInput->GetMagic());
            }
        }
    }
    // When there is only one consumer, we process the boundary tensor as it doesn't have a main subgraph.
    if (tensor->GetConsumers().size() == 1) {
        newBoundaryTensors.push_back(tensor);
        return SUCCESS;
    }

    std::vector<Operation*> subsidiaryConsumers;
    for (const auto& consumer : tensor->GetConsumers()) {
        if (consumer->GetSubgraphID() != mainSubgraphID) {
            subsidiaryConsumers.push_back(consumer);
            APASS_LOG_DEBUG_F(Elements::Operation, "Insert %s[%d] to the subsidiaryConsumers.",
                              consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic());
        }
    }

    // For consumers from other subgraph, we insert a new CONTRACT before them,
    // the intermediate tensor become a new boundary tensor.
    if (subsidiaryConsumers.size() != 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "=========== multi subsidiaryConsumers size: %zu ==========", subsidiaryConsumers.size());
        LogicalTensorPtr newTensor = InsertOpBetween(function, config::GetContractOpcode(), tensor, subsidiaryConsumers,
                                                     mainSubgraphID);
        newBoundaryTensors.push_back(newTensor);
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::ProcessBoundaryTensors(Function& function, LogicalTensors tensors)
{
    for (const auto& tensor : tensors) {
        if (ProcessBoundaryTensor(function, tensor) == FAILED) {
            APASS_LOG_ERROR_F(
                Elements::Tensor,
                "Process boundary tensor failed, tensor magic : %d; Please check the ProcessBoundaryTensor method.",
                tensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::ProcessBoundaryTensor(Function& function, LogicalTensorPtr tensor)
{
    APASS_LOG_INFO_F(Elements::Tensor, "Process boundary tensor, tensor magic : %d, info: %s, mem: %s",
                     tensor->GetMagic(), tensor->Dump().c_str(),
                     BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str());
    // Adapt the producer so that it can write the boundary tensor to DDR.
    if (AdapteTensorProducers(function, tensor) == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "Adapter tensor producer failed, tensor magic : %d; Please check the AdapteTensorProducers method.",
            tensor->GetMagic());
        return FAILED;
    }
    // Adapt the consumer so that it can read the boundary tensor from DDR.
    if (AdapteTensorConsumers(function, tensor) == FAILED) {
        APASS_LOG_ERROR_F(
            Elements::Tensor,
            "Adapter tensor consumer failed, tensor magic : %d; Please check the AdapteTensorConsumers method.",
            tensor->GetMagic());
        return FAILED;
    }
    // Set the boundary tensor's mem type to be DDR
    tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    return SUCCESS;
}

Status IntraSubgraphAdapter::AdapteTensorProducers(Function& function, LogicalTensorPtr tensor)
{
    if (tensor->GetProducers().size() > 1) {
        auto producers = tensor->GetProducers();
        for (Operation* producer : producers) {
            APASS_LOG_DEBUG_F(Elements::Operation, "|---- Producer %s[%d].", producer->GetOpcodeStr().c_str(),
                              producer->GetOpMagic());
            if (producer->GetOpcode() == config::GetContractOpcode()) {
                APASS_LOG_DEBUG_F(Elements::Operation, "|---- Op Attr: %s", producer->Dump().c_str());
            }
            if (OpcodeManager::Inst().GetOpCalcType(producer->GetOpcode()) != OpCalcType::MOVE_OUT &&
                OpcodeManager::Inst().GetOpCalcType(producer->GetOpcode()) != OpCalcType::MOVE_LOCAL) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "OpCalcType error; If the tensor has multiple producers, then the producers can only be "
                    "assemble-family/OP_COPY_OUT.%s",
                    GetFormatBacktrace(*producer).c_str());
                return FAILED;
            }
            if (IsAssembleWithDifferentMemoryType(*producer, MemoryType::MEM_DEVICE_DDR)) {
                SetConfiguredOpcode(*producer, config::GetContractOpcode());
            }
        }
        return SUCCESS;
    }
    if (tensor->GetProducers().size() == 1) {
        Operation* producer = *(tensor->GetProducers().begin());
        APASS_LOG_DEBUG_F(Elements::Operation, "|---- Producer %s[%d].", producer->GetOpcodeStr().c_str(),
                          producer->GetOpMagic());
        if (producer->GetOpcode() == config::GetContractOpcode()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "|---- Op Attr: %s", producer->Dump().c_str());
        }
        if (GraphUtils::IsCrossCoreMoveOp(producer)) {
            producer->SetOpCode(Opcode::OP_COPY_OUT);
            auto convertOpAttr = dynamic_cast<ConvertOpAttribute*>(producer->GetOpAttribute().get());
            if (convertOpAttr != nullptr) {
                std::vector<int64_t> offset(tensor->GetShape().size(), 0);
                producer->SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    convertOpAttr->GetConvertPath().first, OpImmediate::Specified(offset),
                    OpImmediate::Specified(tensor->GetShape()),
                    OpImmediate::Specified(tensor->GetRawTensor()->GetDynRawShape())));
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "change %s[%d] opcode to OP_COPY_OUT.",
                              producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
        }
        if (IsAssembleWithDifferentMemoryType(*producer, MemoryType::MEM_DEVICE_DDR)) {
            SetConfiguredOpcode(*producer, config::GetContractOpcode());
        }
        if (producer->GetOpcode() != config::GetContractOpcode() && producer->GetOpcode() != Opcode::OP_COPY_OUT) {
            InsertOpBetween(function, config::GetContractOpcode(), *producer, tensor);
        }
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Tensor, "Boundary tensor has no producer, tensor magic : %d.", tensor->GetMagic());
    return SUCCESS;
}

static bool IsIndirectView(Operation* op)
{
    if (op->GetOpcode() != config::GetSliceOpcode()) {
        return false;
    }
    if (op->GetOOperands().size() == 0) {
        return false;
    }
    MemoryType toType = op->GetOOperands()[0]->GetMemoryTypeOriginal();
    std::vector<MemoryType> paths;
    Platform::Instance().GetDie().FindNearestPath(MemoryType::MEM_DEVICE_DDR, toType, paths);
    // DDR直达目标内存的路径包含2个节点（起点DDR + 终点），超过2表示需要经过中间层级（如DDR->UB->L1）
    const size_t directPathLength = 2;
    if (paths.size() > directPathLength) {
        return true;
    }
    return false;
}

Status IntraSubgraphAdapter::AdapteTensorConsumers(Function& function, LogicalTensorPtr tensor)
{
    std::unordered_map<int, std::vector<Operation*>> consumerColor2OpsMap;
    for (const auto& consumer : tensor->GetConsumers()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "|---- Consumer %s[%d].", consumer->GetOpcodeStr().c_str(),
                          consumer->GetOpMagic());
        if (consumer->GetOpcode() == config::GetSliceOpcode()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "|---- Op Attr: %s", consumer->Dump().c_str());
        }
        if (GraphUtils::IsCrossCoreMoveOp(consumer)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "%s[%d] : Crosscore move op should not exist at the front of a subgraph.",
                              consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic());
            return FAILED;
        }
        if (IsViewWithDifferentMemoryType(*consumer, MemoryType::MEM_DEVICE_DDR)) {
            SetConfiguredOpcode(*consumer, config::GetSliceOpcode());
            continue;
        }
        if ((consumer->GetOpcode() != config::GetSliceOpcode() && consumer->GetOpcode() != Opcode::OP_COPY_IN) ||
            IsIndirectView(consumer)) {
            consumerColor2OpsMap[consumer->GetSubgraphID()].push_back(consumer);
        }
    }
    for (auto& [color, consumers] : consumerColor2OpsMap) {
        (void)color;
        InsertOpBetween(function, config::GetSliceOpcode(), tensor, consumers);
    }
    return SUCCESS;
}

// For Contract
LogicalTensorPtr IntraSubgraphAdapter::InsertOpBetween(Function& function, Opcode opcode, Operation& op,
                                                       LogicalTensorPtr tensor)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "intraSubgraphAdapter::InsertOpBetween %s 1.",
                      OpcodeManager::Inst().GetOpcodeStr(opcode).c_str());
    ASSERT(opcode == config::GetContractOpcode()) << "[IntraSubgraphAdapter][Operation][ERROR]: Opcode for "
                                                     "IntraSubgraphAdapter::InsertOpBetween is invalid.";
    auto newRawTensor = std::make_shared<RawTensor>(tensor->Datatype(), tensor->GetRawTensor()->rawshape,
                                                    tensor->Format());
    LogicalTensorPtr newTensor = irBuilder_.CreateTensorVar(newRawTensor, tensor->GetOffset(), tensor->GetShape(),
                                                            std::vector<SymbolicScalar>{});
    GraphUtils::CopyDynStatus(newTensor, tensor);
    newTensor->SetMemoryTypeBoth(tensor->GetMemoryTypeOriginal(), true);
    op.ReplaceOutputOperand(tensor, newTensor);
    APASS_LOG_DEBUG_F(Elements::Tensor, "Compare Ori: %s", tensor->Dump().c_str());
    APASS_LOG_DEBUG_F(Elements::Tensor, "Compare New: %s", newTensor->Dump().c_str());

    std::vector<int64_t> offset(tensor->GetShape().size(), 0);
    Operation* newOp = &irBuilder_.CreateTensorOpStmt(function, opcode, {newTensor}, {tensor});
    newOps.push_back(newOp);
    newOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(newTensor->GetMemoryTypeOriginal(), offset,
                                                                tensor->GetDynOffset(), tensor->GetDynValidShape()));
    newOp->SetScopeInfo(op.GetScopeInfo());
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert New Op %s[%d], info: %s", newOp->GetOpcodeStr().c_str(),
                      newOp->GetOpMagic(), newOp->Dump().c_str());
    newOp->UpdateSubgraphID(op.GetSubgraphID());
    newTensor->AddProducer(op);
    newTensor->AddConsumer(newOp);
    tensor->RemoveProducer(op);
    tensor->AddProducer(newOp);
    return newTensor;
}

// For Slice
LogicalTensorPtr IntraSubgraphAdapter::InsertOpBetween(Function& function, Opcode opcode, LogicalTensorPtr tensor,
                                                       const std::vector<Operation*>& ops, int newOpSubgraphID)
{
    /*
        before: tensor -> op
        after: tensor -> newOp -> newTensor ->op
    */
    APASS_LOG_DEBUG_F(Elements::Operation, "IntraSubgraphAdapter::InsertOpBetween %s 2.",
                      OpcodeManager::Inst().GetOpcodeStr(opcode).c_str());
    if (ops.size() == 0) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Insert op between tensor and ops failed; The ops to be inserted can't be empty.");
        return nullptr;
    }
    ASSERT(opcode == config::GetSliceOpcode() || opcode == config::GetContractOpcode())
        << "[IntraSubgraphAdapter][Operation][ERROR]: Opcode for IntraSubgraphAdapter::InsertOpBetween must be "
           "current view-family or assemble-family opcode.";
    auto newRawTensor = std::make_shared<RawTensor>(tensor->Datatype(), tensor->GetRawTensor()->rawshape,
                                                    tensor->Format());
    LogicalTensorPtr newTensor = irBuilder_.CreateTensorVar(newRawTensor, tensor->GetOffset(), tensor->GetShape(),
                                                            tensor->GetDynValidShape());
    newTensor->UpdateOffset(tensor->GetTensorOffset());
    GraphUtils::CopyDynStatus(newTensor, tensor);
    newTensor->SetMemoryTypeBoth(tensor->GetMemoryTypeOriginal(), true);
    APASS_LOG_DEBUG_F(Elements::Tensor, "Compare Ori: %s", tensor->Dump().c_str());
    APASS_LOG_DEBUG_F(Elements::Tensor, "Compare New: %s", newTensor->Dump().c_str());

    for (Operation* op : ops) {
        op->ReplaceInputOperand(tensor, newTensor);
    }

    std::vector<int64_t> offset(tensor->GetShape().size(), 0);
    Operation* newOp = &irBuilder_.CreateTensorOpStmt(function, opcode, {tensor}, {newTensor});
    newOps.push_back(newOp);
    if (opcode == config::GetContractOpcode()) {
        newOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(
            newTensor->GetMemoryTypeOriginal(), offset, tensor->GetDynOffset(), tensor->GetDynValidShape()));
        if (!tensor->GetProducers().empty()) {
            newOp->SetScopeInfo((*tensor->GetProducers().begin())->GetScopeInfo());
        }
    }
    if (opcode == config::GetSliceOpcode()) {
        newOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, newTensor->GetMemoryTypeToBe(),
                                                                tensor->GetDynOffset(), tensor->GetDynValidShape()));
        newOp->SetScopeInfo(ops[0]->GetScopeInfo());
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert New Op %s[%d], info: %s", newOp->GetOpcodeStr().c_str(),
                      newOp->GetOpMagic(), newOp->Dump().c_str());
    newOp->UpdateSubgraphID(newOpSubgraphID == -1 ? ops[0]->GetSubgraphID() : newOpSubgraphID);
    newTensor->AddProducer(newOp);
    for (Operation* op : ops) {
        newTensor->AddConsumer(op);
        tensor->RemoveConsumer(op);
    }
    tensor->AddConsumer(newOp);
    return newTensor;
}

void IntraSubgraphAdapter::CollectProducerColors(LogicalTensorPtr tensor, std::set<int>& colors)
{
    for (const Operation* producer : tensor->GetProducers()) {
        colors.insert(producer->GetSubgraphID());
    }
}

void IntraSubgraphAdapter::CollectConsumerColors(LogicalTensorPtr tensor, std::set<int>& colors)
{
    for (const Operation* consumer : tensor->GetConsumers()) {
        colors.insert(consumer->GetSubgraphID());
    }
}

std::set<int> IntraSubgraphAdapter::SetIntersection(std::set<int>& a, std::set<int>& b)
{
    std::set<int> c;
    for (auto i : a) {
        if (b.find(i) != b.end()) {
            c.insert(i);
        }
    }
    return c;
}

LogicalTensors IntraSubgraphAdapter::CollectBoundaryTensors(Function& function)
{
    LogicalTensors ret;
    TensorSet allTensors = GraphUtils::GetAllTensors(function);
    for (const auto& tensor : allTensors) {
        if (!tensor)
            continue;
        std::set<int> colors;
        CollectProducerColors(tensor, colors);
        CollectConsumerColors(tensor, colors);
        if (colors.size() > 1) {
            ret.push_back(tensor);
        }
    }
    return ret;
}

Status IntraSubgraphAdapter::PostCheck(Function& function)
{
    IntraSubgraphAdapterChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace npu::tile_fwk
