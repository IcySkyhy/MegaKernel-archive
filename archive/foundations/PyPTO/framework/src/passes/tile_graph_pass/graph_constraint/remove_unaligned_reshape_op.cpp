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
 * \file remove_unaligned_reshape_op.cpp
 * \brief
 */

#include "remove_unaligned_reshape_op.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/alignment_utils.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveUnalignedReshape"

namespace npu::tile_fwk {
/*
before:
    add->reshape(padded)->mul

after:
    add->copyout->reshape->copyin->mul
*/
static void SetReshapeCopyOutValidShapeAttr(Operation& op)
{
    if (op.GetOOperands().empty()) {
        return;
    }
    auto output = op.GetOOperands().front();
    auto dynValidShape = output->GetDynValidShape();
    if (dynValidShape.empty()) {
        return;
    }
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    if (copyAttr != nullptr) {
        copyAttr->SetToDynValidShape(OpImmediate::Specified(dynValidShape));
    }
}

static void SetReshapeCopyInValidShapeAttr(Operation& op)
{
    if (op.GetIOperands().empty()) {
        return;
    }
    auto input = op.GetIOperands().front();
    auto dynValidShape = input->GetDynValidShape();
    if (dynValidShape.empty()) {
        return;
    }
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    if (copyAttr != nullptr) {
        copyAttr->SetFromDynValidShape(OpImmediate::Specified(dynValidShape));
    }
}

Status RemoveUnalignedReshape::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveUnalignedReshape.");
    ReplaceDynUnalignedReshapeOps(function);
    CollectReshapeOps(function);
    for (auto& a : copyOuts) {
        GraphUtils::CopyDynStatus(a.output, a.input);
        auto& newCopyOut = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_COPY_OUT, {a.input}, {a.output});
        newOps.push_back(&newCopyOut);
        newCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            a.from, OpImmediate::Specified(a.toOffset), OpImmediate::Specified(newCopyOut.iOperand.front()->shape),
            OpImmediate::Specified(newCopyOut.oOperand.front()->tensor->GetDynRawShape())));
        auto producerOp = *(a.input->GetProducers().begin());
        newCopyOut.UpdateSubgraphID(producerOp->GetSubgraphID());
        APASS_LOG_INFO_F(Elements::Operation,
                         "ADD OP_COPY_OUT, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.",
                         newCopyOut.opmagic, a.input->magic, a.output->magic);
    }
    for (auto& b : copyIns) {
        GraphUtils::CopyDynStatus(b.input, b.output);
        auto& newCopyIn = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_COPY_IN, {b.input}, {b.output});
        newOps.push_back(&newCopyIn);
        newCopyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(b.fromOffset), b.to, OpImmediate::Specified(newCopyIn.oOperand.front()->shape),
            OpImmediate::Specified(newCopyIn.iOperand.front()->tensor->GetDynRawShape()),
            OpImmediate::Specified(newCopyIn.iOperand.front()->GetDynValidShape())));
        auto consumerOp = *(b.output->GetConsumers().begin());
        newCopyIn.UpdateSubgraphID(consumerOp->GetSubgraphID());
        APASS_LOG_INFO_F(Elements::Operation,
                         "ADD OP_VIEW, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.", newCopyIn.opmagic,
                         b.input->magic, b.output->magic);
    }
    if (!newOps.empty()) {
        if (InferShapeUtils::InferShape(function, newOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
            return FAILED;
        }
    }

    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveUnalignedReshape.");
    return SUCCESS;
}

LogicalTensorPtr RemoveUnalignedReshape::InsertIOTensor(
    Function& function, Operation& op, std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>>& rawIO,
    LogicalTensorPtr& ioTensor)
{
    (void)function;
    (void)op;
    if (rawIO.count(ioTensor->tensor->rawmagic) == 0) {
        auto reshapeRawTensor = std::make_shared<RawTensor>(ioTensor->Datatype(), ioTensor->shape, ioTensor->Format());
        rawIO.insert({ioTensor->tensor->rawmagic, reshapeRawTensor});
    }
    auto newReshapeIO = irBuilder_.CreateTensorVar(rawIO[ioTensor->tensor->rawmagic], ioTensor->offset, ioTensor->shape,
                                                   std::vector<SymbolicScalar>{});
    newReshapeIO->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    return newReshapeIO;
}

bool RemoveUnalignedReshape::CheckUnaligned(Operation& op)
{
    int lastIdx;
    for (const auto& input : op.GetIOperands()) {
        if (input != nullptr && input->tensor != nullptr) {
            lastIdx = input->shape.size() - 1;
            if (input->shape[lastIdx] != input->tensor->rawshape[lastIdx]) {
                return true;
            }
        }
    }
    for (const auto& output : op.GetOOperands()) {
        if (output != nullptr && output->tensor != nullptr) {
            lastIdx = output->shape.size() - 1;
            if (output->shape[lastIdx] != output->tensor->rawshape[lastIdx]) {
                return true;
            }
        }
    }
    return false;
}

std::vector<int64_t> FindChangedDims(const std::vector<int64_t>& inputShapes, const std::vector<int64_t>& outputShapes)
{
    int inputDimSize = inputShapes.size();
    int outputDimSize = outputShapes.size();

    int left = -1;
    int right = -1;
    std::vector<int64_t> changedInputAxes = {};

    for (int i = 0; i < std::min(inputDimSize, outputDimSize); ++i) {
        if (inputShapes[i] != outputShapes[i] && left == -1) {
            left = i; // left第一次shape不等的位置
        }

        if (inputShapes[inputDimSize - 1 - i] != outputShapes[outputDimSize - 1 - i] && right == -1) {
            right = inputDimSize - 1 - i; // right第一次shape不等的位置
        }
    }

    if (left <= right && left != -1 && right != -1) {
        for (int i = left; i <= right; ++i) {
            changedInputAxes.push_back(i);
        }
    }

    return changedInputAxes;
}
void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOps(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start ReplaceDynUnalignedReshapeOps.");
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op->GetOpcode() != Opcode::OP_RESHAPE || processedReshapeOps.count(op->GetOpMagic())) {
            continue;
        }
        auto input = op->GetIOperands().front();
        auto output = op->GetOOperands().front();
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
            output->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            ReplaceDynUnalignedReshapeOpsForUB(function, *op);
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                   output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            ReplaceDynUnalignedReshapeOpsForDDR(function, *op);
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End ReplaceDynUnalignedReshapeOps.");
}

void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOpsForUB(Function& function, Operation& op)
{
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto inputShapes = input->shape;
    auto outputShapes = output->shape;
    auto changedDims = FindChangedDims(outputShapes, inputShapes);

    auto inDynValidShape = input->GetDynValidShape();
    auto outDynValidShape = output->GetDynValidShape();

    for (const auto& dim : changedDims) {
        if ((size_t)dim >= outDynValidShape.size()) {
            APASS_LOG_WARN_F(Elements::Operation, "The dynValidShape of output[%d] of op[%d] has no [%ld] index.",
                             output->GetMagic(), op.GetOpMagic(), static_cast<long>(dim));
            break;
        } else if (!outDynValidShape[dim].IsImmediate()) {
            auto tmpWorkSpaceIn = irBuilder_.CreateTensorVar(input->Datatype(), input->shape,
                                                             std::vector<SymbolicScalar>{}, input->Format());
            auto tmpWorkSpaceOut = irBuilder_.CreateTensorVar(input->Datatype(), output->shape,
                                                              std::vector<SymbolicScalar>{}, output->Format());

            tmpWorkSpaceIn->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            tmpWorkSpaceIn->UpdateDynValidShape(inDynValidShape);
            tmpWorkSpaceOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            tmpWorkSpaceOut->UpdateDynValidShape(outDynValidShape);

            auto& reshapeCopyOutOp = PassOperationUtils::AddOperation(
                function, Opcode::OP_RESHAPE_COPY_OUT, {input}, {tmpWorkSpaceIn}, [&input](Operation& newOp) {
                    newOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                        MemoryType::MEM_UB, OpImmediate::Specified(std::vector<SymbolicScalar>(input->shape.size(), 0)),
                        OpImmediate::Specified(input->shape), OpImmediate::Specified(input->tensor->GetDynRawShape()),
                        OpImmediate::Specified(input->GetDynValidShape())));
                });
            op.ReplaceInput(tmpWorkSpaceIn, op.GetIOperands().front());
            op.ReplaceOutput(tmpWorkSpaceOut, op.GetOOperands().front());
            auto& reshapeCopyInOp = PassOperationUtils::AddOperation(
                function, Opcode::OP_RESHAPE_COPY_IN, {tmpWorkSpaceOut}, {output}, [&output](Operation& newOp) {
                    newOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                        OpImmediate::Specified(std::vector<SymbolicScalar>(output->shape.size(), 0)),
                        MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(output->shape),
                        OpImmediate::Specified(output->tensor->GetDynRawShape()),
                        OpImmediate::Specified(output->GetDynValidShape())));
                });
            newOps.push_back(&reshapeCopyOutOp);
            newOps.push_back(&reshapeCopyInOp);

            reshapeCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
            reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());

            SetReshapeCopyOutValidShapeAttr(reshapeCopyOutOp);
            SetReshapeCopyInValidShapeAttr(reshapeCopyInOp);
            APASS_LOG_INFO_F(Elements::Operation,
                             "Reshape op %d is replaceed by reshapeCopyOutOp %d and reshapeCopyInOp %d.", op.opmagic,
                             reshapeCopyOutOp.opmagic, reshapeCopyInOp.opmagic);
            processedReshapeOps.insert(op.GetOpMagic());
            function.EraseOperations(true, false);
            break;
        }
    }
}

Operation* RemoveUnalignedReshape::CopyBranchBetweenCopyOut2Reshape(
    Function& function, const std::vector<std::pair<Operation*, LogicalTensorPtr>>& toCopyProducerTensor,
    const int& consumerIndex)
{
    bool canToCopy = false;
    Operation* branchOp = nullptr;
    LogicalTensorPtr curTensor = nullptr;
    LogicalTensorPtr preTensor = nullptr;
    LogicalTensorPtr preCloneTensor = nullptr;
    Operation* preOp = nullptr;
    for (auto it = toCopyProducerTensor.rbegin(); it != toCopyProducerTensor.rend(); ++it) {
        auto producerTensor = *it;
        auto tensor = producerTensor.second;
        curTensor = tensor->Clone(function, true);
        if (!canToCopy && tensor->GetConsumers().size() > 1) {
            branchOp = *(std::next(tensor->GetConsumers().begin(), consumerIndex));
            canToCopy = true;
            branchOp->ReplaceInput(curTensor, tensor);
        }
        if (canToCopy) {
            if (preOp != nullptr) {
                auto& newOp = preOp->CloneOperation(function, preOp->GetIOperands(), preOp->GetOOperands());
                newOps.push_back(&newOp);
                newOp.UpdateSubgraphID(preOp->GetSubgraphID());
                newOp.ReplaceInput(curTensor, tensor);
                newOp.ReplaceOutput(preCloneTensor, preTensor);
            }
            if (!tensor->GetProducers().empty()) {
                preOp = producerTensor.first;
            }
        }
        preTensor = tensor;
        preCloneTensor = curTensor;
    }
    // 此时preOp 为 copyOutop preTensor为CopyOut的输出
    Operation* newCopyOutOp = &(preOp->CloneOperation(function, preOp->GetIOperands(), preOp->GetOOperands()));
    newOps.push_back(newCopyOutOp);
    newCopyOutOp->UpdateSubgraphID(preOp->GetSubgraphID());
    newCopyOutOp->ReplaceOutput(preCloneTensor, preTensor);
    DeadOperationEliminator::EliminateDeadOperation(function);
    return newCopyOutOp;
}

LogicalTensorPtr RemoveUnalignedReshape::HandleNoCopyOutInProducer(Function& function, Operation& op,
                                                                   bool& checkOverUbSize)
{
    auto input = op.GetIOperands().front();
    auto copyShape = input->GetShape();
    auto copyRawShape = input->tensor->GetDynRawShape();
    auto copyDynShape = input->GetDynValidShape();
    Offset offset(copyShape.size(), 0);

    auto copyInOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, copyDynShape);
    copyInOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    // 为copy到Ub的Tensor进行32B对齐
    AlignmentUtils::ProcessLastDim32BAlignedOnUB(copyInOutputPtr);

    // 要copy到UB的Tensor，进行32B对齐之后判断是否超UB
    const int UB_SIZE_THRESHOLD = static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB));
    if (copyInOutputPtr->GetDataSize() > UB_SIZE_THRESHOLD) {
        APASS_LOG_WARN_F(Elements::Tensor,
                         "Tensor [%d] can not copy to UB, tensor size [%ld] exceeds the UB size [%d] limit.",
                         input->magic, input->GetDataSize(), UB_SIZE_THRESHOLD);
        checkOverUbSize = true;
        return nullptr;
    }
    auto& copyInOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_IN, {input}, {copyInOutputPtr},
        [&input, &copyShape, &copyRawShape, &copyDynShape](Operation& newOp) {
            newOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(input->GetOffset()), MemoryType::MEM_UB, OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    newOps.push_back(&copyInOp);
    copyInOp.UpdateSubgraphID(op.GetSubgraphID());

    auto copyOutOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, copyDynShape);
    copyOutOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto& copyOutOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_OUT, {copyInOutputPtr}, {copyOutOutputPtr},
        [&offset, &copyShape, &copyRawShape, &copyDynShape](Operation& newOp) {
            newOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                MemoryType::MEM_UB, OpImmediate::Specified(offset), OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    newOps.push_back(&copyOutOp);
    copyOutOp.UpdateSubgraphID(op.GetSubgraphID());

    op.ReplaceInput(copyOutOutputPtr, input);
    return copyOutOutputPtr;
}

int RemoveUnalignedReshape::FindConsumerIndex(LogicalTensorPtr input, Operation* consumerOp)
{
    int index = 0;
    for (auto con : input->GetConsumers()) {
        if (con->GetOpMagic() == consumerOp->GetOpMagic()) {
            return index;
        }
        index++;
    }
    return -1;
}

void RemoveUnalignedReshape::GetPathBetweenSingleCopyOutAndReshape(
    Operation* op, std::vector<std::pair<Operation*, LogicalTensorPtr>>& toCopyProducerTensor, bool& findCopyOut,
    bool& needToCopy, int& index)
{
    for (auto input : op->GetIOperands()) {
        auto producers = input->GetProducers();
        if (producers.empty()) {
            return;
        }
        for (auto producerOp : producers) {
            auto opcode = producerOp->GetOpcode();
            if (opcode == Opcode::OP_COPY_OUT) {
                std::pair<Operation*, LogicalTensorPtr> producerTensor = std::make_pair(producerOp, input);
                toCopyProducerTensor.push_back(producerTensor);
                if (input->GetConsumers().size() > 1) {
                    needToCopy = true;
                    index = FindConsumerIndex(input, op);
                }
                findCopyOut = true;
                return;
            }

            // 其他类型的op（包括view/assemble或其他op），继续向前追溯
            GetPathBetweenSingleCopyOutAndReshape(producerOp, toCopyProducerTensor, findCopyOut, needToCopy, index);
            if (findCopyOut) {
                std::pair<Operation*, LogicalTensorPtr> producerTensor = std::make_pair(producerOp, input);
                toCopyProducerTensor.push_back(producerTensor);
                if (input->GetConsumers().size() > 1) {
                    needToCopy = true;
                    index = FindConsumerIndex(input, op);
                }
                return;
            }
        }
    }
}

void RemoveUnalignedReshape::InsertReshapeCopy(Function& function, Operation& op)
{
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    // 进行处理前判断，防止误修改
    std::vector<Operation*> copyOutOps;
    Operation* copyOutOp = nullptr;
    std::vector<std::pair<Operation*, LogicalTensorPtr>> toCopyProducerTensor;
    int index = -1;
    bool findCopyOut = false;
    bool needToCopy = false;
    bool checkOverUbSize = false;
    // index表示copyout到reshape之间，有多个消费者的Tensor的第几个消费者是包含需要处理的reshape的
    FindAllProducerCopyOuts(input, copyOutOps);
    if (copyOutOps.empty()) {
        auto newReshapeIo = HandleNoCopyOutInProducer(function, op, checkOverUbSize);
        if (!checkOverUbSize && newReshapeIo != nullptr) {
            copyOutOp = *(newReshapeIo->GetProducers().begin());
        }
    } else if (copyOutOps.size() > 1) {
        checkOverUbSize = !ProcessMultipleCopyOuts(copyOutOps);
    } else {
        copyOutOp = copyOutOps.front();
        GetPathBetweenSingleCopyOutAndReshape(&op, toCopyProducerTensor, findCopyOut, needToCopy, index);
    }
    std::vector<Operation*> copyInOps;
    if (checkNonCopyInConsumerExists(output, copyInOps)) {
        HandleNoCopyInConsumer(function, op, output, copyInOps, checkOverUbSize);
    }

    // 进行处理
    if (!checkOverUbSize) {
        if (needToCopy) {
            copyOutOp = CopyBranchBetweenCopyOut2Reshape(function, toCopyProducerTensor, index);
        }
        CollectSiblingCopyInConsumersOfDDRReshapeInput(input, op, copyInOps);
        if (copyOutOp != nullptr) {
            ProcessCopyOutOfDDRReshape(copyOutOp);
        }
        ProcessCopyInOfDDRReshape(copyInOps);
    } else {
        APASS_LOG_WARN_F(Elements::Tensor,
                         "Reshape[%d] on GM had processed failed, "
                         "because the size of input[%d] or output[%d] of reshape[%d] exceeded ub if copy to ub.",
                         op.GetOpMagic(), input->GetMagic(), output->GetMagic(), op.GetOpMagic());
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Reshape[%d] on GM had processed successfully.", op.GetOpMagic());
    processedReshapeOps.insert(op.GetOpMagic());
}

void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOpsForDDR(Function& function, Operation& op)
{
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto outputValidShape = output->GetDynValidShape();

    bool hasNonImmediate = false;
    auto changedDims = FindChangedDims(output->shape, input->shape);
    for (const auto& dim : changedDims) {
        if ((size_t)dim >= outputValidShape.size()) {
            APASS_LOG_WARN_F(Elements::Operation, "The dynValidShape of output[%d] of op[%d] has no [%ld] index.",
                             output->GetMagic(), op.GetOpMagic(), static_cast<long>(dim));
            break;
        } else if (!outputValidShape[dim].IsImmediate()) {
            hasNonImmediate = true;
            break;
        }
    }
    if (hasNonImmediate) {
        InsertReshapeCopy(function, op);
    }
}

void RemoveUnalignedReshape::HandleNoCopyInConsumer(Function& function, Operation& op, LogicalTensorPtr output,
                                                    std::vector<Operation*>& copyInOps, bool& checkOverUbSize)
{
    auto newCopyinTensorPtr = irBuilder_.CreateTensorVar(output->Datatype(), output->GetShape(),
                                                         output->GetDynValidShape());
    newCopyinTensorPtr->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    AlignmentUtils::ProcessLastDim32BAlignedOnUB(newCopyinTensorPtr);

    // 要copy到UB的Tensor，进行32B对齐之后判断是否超UB
    const size_t UB_SIZE_THRESHOLD = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    if (static_cast<size_t>(newCopyinTensorPtr->GetDataSize()) > UB_SIZE_THRESHOLD) {
        APASS_LOG_WARN_F(
            Elements::Tensor,
            "The size[%ld] of copyTensor[%d] from output of reshape op[%d] should not exceed %zu after padding. "
            "Consider reducing its size.",
            newCopyinTensorPtr->GetDataSize(), newCopyinTensorPtr->GetMagic(), op.GetOpMagic(), UB_SIZE_THRESHOLD);
        checkOverUbSize = true;
        return;
    }
    auto& newCopyInOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_IN, {output}, {newCopyinTensorPtr}, [&output](Operation& newOp) {
            newOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(std::vector<SymbolicScalar>(output->GetShape().size(), 0)), MemoryType::MEM_UB,
                OpImmediate::Specified(output->GetShape()), OpImmediate::Specified(output->tensor->GetDynRawShape()),
                OpImmediate::Specified(output->GetDynValidShape())));
        });
    newOps.push_back(&newCopyInOp);
    newCopyInOp.UpdateSubgraphID(op.GetSubgraphID());

    auto newCopyoutTensorPtr = irBuilder_.CreateTensorVar(output->Datatype(), output->GetShape(),
                                                          output->GetDynValidShape());
    newCopyoutTensorPtr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto& newCopyOutOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_OUT, {newCopyinTensorPtr}, {newCopyoutTensorPtr}, [&output](Operation& newOp) {
            newOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                MemoryType::MEM_UB, OpImmediate::Specified(std::vector<SymbolicScalar>(output->GetShape().size(), 0)),
                OpImmediate::Specified(output->GetShape()), OpImmediate::Specified(output->tensor->GetDynRawShape()),
                OpImmediate::Specified(output->GetDynValidShape())));
        });
    newOps.push_back(&newCopyOutOp);
    newCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
    auto consumers = output->GetConsumers();
    std::vector<Operation*> reshapeConsumers;
    for (auto& consumer : consumers) {
        reshapeConsumers.push_back(consumer);
    }
    for (auto& consumer : reshapeConsumers) {
        if (consumer->GetOpcode() == Opcode::OP_COPY_IN) {
            continue;
        }
        consumer->ReplaceInput(newCopyoutTensorPtr, output);
        output->RemoveConsumer(consumer);
    }
    copyInOps.push_back(&newCopyInOp);
}

bool RemoveUnalignedReshape::CheckAllCopyOutInputsNonUb(const std::vector<Operation*>& copyOutOps)
{
    for (auto* copyOutOp : copyOutOps) {
        auto copyOutInput = copyOutOp->GetIOperands().front();
        auto copyOutInputMemType = copyOutInput->GetMemoryTypeOriginal();
        if (copyOutInputMemType == MemoryType::MEM_UB) {
            return false;
        }
    }
    return true;
}

void RemoveUnalignedReshape::CollectSiblingCopyInConsumersOfDDRReshapeInput(LogicalTensorPtr reshapeInput,
                                                                            Operation& reshapeOp,
                                                                            std::vector<Operation*>& copyInOps)
{
    auto consumers = reshapeInput->GetConsumers();
    for (auto* consumerOp : consumers) {
        if (consumerOp == nullptr || consumerOp->GetOpMagic() == reshapeOp.GetOpMagic() ||
            consumerOp->GetOpcode() != Opcode::OP_COPY_IN || consumerOp->GetIOperands().empty() ||
            consumerOp->GetOOperands().empty()) {
            continue;
        }
        auto copyInOutput = consumerOp->GetOOperands().front();
        if (reshapeInput == nullptr || copyInOutput == nullptr ||
            reshapeInput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        if (copyInOutput->GetMemoryTypeOriginal() == MemoryType::MEM_UB ||
            copyInOutput->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            copyInOps.push_back(consumerOp);
        }
    }
}

bool RemoveUnalignedReshape::ProcessMultipleCopyOuts(std::vector<Operation*>& copyOutOps)
{
    bool allCopyOutInputsNonUb = CheckAllCopyOutInputsNonUb(copyOutOps);
    if (allCopyOutInputsNonUb) {
        auto* firstCopyOutOp = copyOutOps.front();
        return ProcessCopyOutOfDDRReshape(firstCopyOutOp);
    }

    for (auto* cOp : copyOutOps) {
        if (!ProcessCopyOutOfDDRReshape(cOp)) {
            return false;
        }
    }
    return true;
}

bool RemoveUnalignedReshape::ProcessCopyOutOfDDRReshape(Operation* copyOutOp)
{
    // 当copyout的输入是ub输出为ddr可以直接转化为reshapecopyop
    // 否则需要插copy
    auto copyOutInput = copyOutOp->GetIOperands().front();
    auto copyOutInputMemType = copyOutInput->GetMemoryTypeOriginal();
    auto copyOutOutput = copyOutOp->GetOOperands().front();
    auto copyOutOutputMemType = copyOutOutput->GetMemoryTypeOriginal();
    if (copyOutInputMemType == MemoryType::MEM_UB && copyOutOutputMemType == MemoryType::MEM_DEVICE_DDR) {
        copyOutOp->SetOpCode(Opcode::OP_RESHAPE_COPY_OUT);
        SetReshapeCopyOutValidShapeAttr(*copyOutOp);
        return true;
    } else if (copyOutInputMemType == MemoryType::MEM_L0C && copyOutOutputMemType == MemoryType::MEM_DEVICE_DDR) {
        copyOutOp->SetOpCode(Opcode::OP_L0C_RESHAPE_COPY_OUT);
        SetReshapeCopyOutValidShapeAttr(*copyOutOp);
        return true;
    }
    return true; // Default return for other cases
}

void RemoveUnalignedReshape::ProcessCopyInOfDDRReshape(std::vector<Operation*>& copyInOps)
{
    for (auto* copyInOp : copyInOps) {
        auto copyInInput = copyInOp->GetIOperands().front();
        auto copyInOutput = copyInOp->GetOOperands().front();
        auto copyInOutputMemType = copyInOutput->GetMemoryTypeOriginal();

        if (copyInInput->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            if (copyInOutputMemType == MemoryType::MEM_UB) {
                copyInOp->SetOpCode(Opcode::OP_RESHAPE_COPY_IN);
                SetReshapeCopyInValidShapeAttr(*copyInOp);
            } else if (copyInOutputMemType == MemoryType::MEM_L1) {
                copyInOp->SetOpCode(Opcode::OP_L1_RESHAPE_COPY_IN);
                SetReshapeCopyInValidShapeAttr(*copyInOp);
            }
        }
    }
}

// 寻找reshape输入的所有copyout
void RemoveUnalignedReshape::FindAllProducerCopyOuts(LogicalTensorPtr tensor, std::vector<Operation*>& copyOutOps)
{
    auto producers = tensor->GetProducers();
    if (producers.empty()) {
        return;
    }
    for (auto producerOp : producers) {
        auto opcode = producerOp->GetOpcode();
        if (opcode == Opcode::OP_COPY_OUT) {
            copyOutOps.push_back(producerOp);
            continue;
        }
        if (opcode == Opcode::OP_RESHAPE_COPY_OUT) {
            continue;
        }

        // 其他类型的op（包括view/assemble或其他op），继续向前追溯
        auto inputOperands = producerOp->GetIOperands();
        for (auto input : inputOperands) {
            FindAllProducerCopyOuts(input, copyOutOps);
        }
    }
}

/* 从tensor的消费者列表中查找OP_COPY_IN，如果未找到OP_COPY_IN，
 * 则标记hasViewOrAssemble为true，表示需要插入拷贝。
 */
bool RemoveUnalignedReshape::checkNonCopyInConsumerExists(LogicalTensorPtr tensor, std::vector<Operation*>& copyInOps)
{
    bool hasNoCopyInConsumer = false;
    auto consumers = tensor->GetConsumers();
    for (auto* consumerOp : consumers) {
        auto opcode = consumerOp->GetOpcode();
        if (opcode == Opcode::OP_COPY_IN) {
            copyInOps.push_back(consumerOp);
        } else {
            hasNoCopyInConsumer = true;
        }
    }
    return hasNoCopyInConsumer;
}

void RemoveUnalignedReshape::CollectReshapeOps(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE || processedReshapeOps.count(op.GetOpMagic())) {
            continue;
        }

        if (!CheckUnaligned(op)) {
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if ((input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) ||
            (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB)) {
            continue;
        }

        // 插入copyout
        LogicalTensorPtr newReshapeInput = InsertIOTensor(function, op, reshapeRawInputs, input);
        copyOuts.emplace_back(
            CopyOutOpMemUnalign{input->GetMemoryTypeOriginal(), input->offset, input, newReshapeInput});
        op.ReplaceInput(newReshapeInput, input);

        // 插入copyin
        LogicalTensorPtr newReshapeOutput = InsertIOTensor(function, op, reshapeRawOutputs, output);
        copyIns.emplace_back(
            CopyInOpMemUnalign{output->GetMemoryTypeOriginal(), output->offset, newReshapeOutput, output});
        op.ReplaceOutput(newReshapeOutput, output);
        output->tensor->actualRawmagic = -1;
        op.GetOOperands().front()->tensor->actualRawmagic = op.GetIOperands().front()->tensor->GetRawMagic();
        processedReshapeOps.insert(op.GetOpMagic());
    }
}

} // namespace npu::tile_fwk
