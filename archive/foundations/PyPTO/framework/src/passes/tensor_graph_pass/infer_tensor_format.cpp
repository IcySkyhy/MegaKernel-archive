/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_tensor_format.cpp
 * \brief
 */

#include "infer_tensor_format.h"

#include "interface/operation/conv/conv_vec_tile_inference.h"
#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/platform.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk_op.h"

#define MODULE_NAME "InferTensorFormat"

namespace npu::tile_fwk {

// =============================================================================
// Op 分类
// =============================================================================

const std::unordered_set<Opcode> InferTensorFormat::kPassThroughOps = {Opcode::OP_VIEW, Opcode::OP_VIEW_TYPE,
                                                                       Opcode::OP_RESHAPE};

// =============================================================================
// Format 查询
// =============================================================================

TileOpFormat InferTensorFormat::GetRequiredInputFormat(Opcode opcode, const std::string& arch, size_t inputPos)
{
    const auto& formatList = OpcodeManager::Inst().GetSupportOpFormatList(opcode);
    auto archIt = formatList.find(arch);
    if (archIt == formatList.end() || archIt->second.empty()) {
        return TileOpFormat::TILEOP_ND;
    }
    const auto& inputFormats = archIt->second[0];
    if (inputPos >= inputFormats.size()) {
        return TileOpFormat::TILEOP_ND;
    }
    return inputFormats[inputPos];
}

TileOpFormat InferTensorFormat::GetOutputFormat(Opcode opcode, const std::string& arch, size_t outputPos)
{
    constexpr size_t kOutputFormatListIndex = 1; // formatList[0]=输入格式, [1]=输出格式
    const auto& formatList = OpcodeManager::Inst().GetSupportOpFormatList(opcode);
    auto archIt = formatList.find(arch);
    if (archIt == formatList.end() || archIt->second.size() <= kOutputFormatListIndex) {
        return TileOpFormat::TILEOP_ND;
    }
    const auto& outputFormats = archIt->second[kOutputFormatListIndex];
    if (outputPos >= outputFormats.size()) {
        return TileOpFormat::TILEOP_ND;
    }
    return outputFormats[outputPos];
}

int InferTensorFormat::GetOpGroupValue(Operation* op)
{
    if (op != nullptr && op->HasAttr(CONV_GROUPS_ATTR)) {
        return static_cast<int>(op->GetIntAttribute(CONV_GROUPS_ATTR));
    }
    return 1;
}

int InferTensorFormat::GetTransDataGroupValue(const std::shared_ptr<LogicalTensor>& srcTensor,
                                              const std::shared_ptr<LogicalTensor>& fakeDstTensor, Operation* relatedOp)
{
    int group = GetOpGroupValue(relatedOp);
    if (group != 1 || srcTensor == nullptr) {
        return group;
    }
    for (auto* producer : srcTensor->GetProducers()) {
        group = GetOpGroupValue(producer);
        if (group != 1) {
            return group;
        }
    }
    if (srcTensor != fakeDstTensor) {
        for (auto* consumer : fakeDstTensor->GetConsumers()) {
            group = GetOpGroupValue(consumer);
            if (group != 1) {
                return group;
            }
        }
    }
    return 1;
}

bool InferTensorFormat::IsSupportedTransData(TileOpFormat srcFormat, TileOpFormat targetFormat)
{
    if (srcFormat == targetFormat) {
        return true;
    }
    if (srcFormat == TileOpFormat::TILEOP_ND) {
        return targetFormat == TileOpFormat::TILEOP_NC1HWC0 || targetFormat == TileOpFormat::TILEOP_FRACTAL_Z ||
               targetFormat == TileOpFormat::TILEOP_NDC1HWC0 || targetFormat == TileOpFormat::TILEOP_FRACTAL_Z_3D;
    }
    return (srcFormat == TileOpFormat::TILEOP_NC1HWC0 || srcFormat == TileOpFormat::TILEOP_NDC1HWC0) &&
           targetFormat == TileOpFormat::TILEOP_ND;
}

bool InferTensorFormat::IsNdNzCompatibleFormat(TileOpFormat srcFormat, TileOpFormat targetFormat)
{
    return (srcFormat == TileOpFormat::TILEOP_ND && targetFormat == TileOpFormat::TILEOP_NZ) ||
           (srcFormat == TileOpFormat::TILEOP_NZ && targetFormat == TileOpFormat::TILEOP_ND);
}

bool InferTensorFormat::IsValidTileOpFormat(int64_t format)
{
    return format >= 0 && format < static_cast<int64_t>(TileOpFormat::TILEOP_FORMAT_NUM);
}

// =============================================================================
// 图操作
// =============================================================================

bool InferTensorFormat::IsFunctionOutcast(const Function& function, const std::shared_ptr<LogicalTensor>& tensor)
{
    for (const auto& outcast : function.GetOutcast()) {
        if (outcast == tensor) {
            return true;
        }
    }
    return false;
}

int InferTensorFormat::FindInputPosition(const Operation& op, const std::shared_ptr<LogicalTensor>& tensor)
{
    const auto& inputs = op.GetIOperands();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == tensor) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

Operation* InferTensorFormat::FindRelatedConvOp(const std::shared_ptr<LogicalTensor>& srcTensor,
                                                const std::shared_ptr<LogicalTensor>& fakeDstTensor,
                                                TileOpFormat targetFormat)
{
    if (targetFormat == TileOpFormat::TILEOP_ND) {
        if (srcTensor == nullptr) {
            return nullptr;
        }
        for (auto* producer : srcTensor->GetProducers()) {
            if (producer != nullptr &&
                (producer->GetOpcode() == Opcode::OP_CONV2D || producer->GetOpcode() == Opcode::OP_CONV3D)) {
                return producer;
            }
        }
    } else {
        if (fakeDstTensor == nullptr) {
            return nullptr;
        }
        for (auto* consumer : fakeDstTensor->GetConsumers()) {
            if (consumer != nullptr &&
                (consumer->GetOpcode() == Opcode::OP_CONV2D || consumer->GetOpcode() == Opcode::OP_CONV3D)) {
                return consumer;
            }
        }
    }
    return nullptr;
}

void InferTensorFormat::ApplyTransDataVecTile(const std::shared_ptr<LogicalTensor>& srcTensor,
                                              TileOpFormat targetFormat, Operation* relatedOp)
{
    ASSERT(OperationErr::OP_NULL_POINTER, relatedOp != nullptr)
        << "Null relatedOp when applying TransData vec tile for src tensor [" << srcTensor->GetMagic()
        << "], target format " << std::to_string(targetFormat) << ".";
    int64_t c0 = srcTensor->Datatype() == DataType::DT_FP32 ? 8 : 16;
    TileOpFormat srcFormat = srcTensor->GetRawTensor()->format;
    VecTile oriVectile = relatedOp->GetTileShape().GetVecTile();
    ASSERT(DistributedErrorCode::INVALID_TILE_SHAPE, oriVectile.tile.back() % c0 == 0)
        << "The last dimension of `tile_shape` should be 32-byte aligned.";
    if (oriVectile.tile.size() == NUM3) {
        oriVectile.tile.insert(oriVectile.tile.begin() + NUM2, 1);
        TileShape opTileShape = relatedOp->GetTileShape();
        opTileShape.SetVecTile(oriVectile);
        relatedOp->UpdateTileShape(opTileShape);
    }
    if (srcFormat == TileOpFormat::TILEOP_NC1HWC0 && targetFormat == TileOpFormat::TILEOP_ND) {
        VecTile tmpVectile = relatedOp->GetTileShape().GetVecTile();
        tmpVectile.tile[1] = tmpVectile.tile[1] / c0;
        tmpVectile.tile.emplace_back(c0);
        TileShape::Current().SetVecTile(tmpVectile);
    } else if (srcFormat == TileOpFormat::TILEOP_NDC1HWC0 && targetFormat == TileOpFormat::TILEOP_ND) {
        VecTile tmpVectile = relatedOp->GetTileShape().GetVecTile();
        tmpVectile.tile[0] = 1;
        std::swap(tmpVectile.tile[1], tmpVectile.tile[2]);
        tmpVectile.tile[2] = tmpVectile.tile[2] / c0;
        tmpVectile.tile.emplace_back(c0);
        tmpVectile.tile[1] *= NUM4;
        tmpVectile.tile[3] *= NUM4;
        TileShape::Current().SetVecTile(tmpVectile);
    } else if (srcFormat == TileOpFormat::TILEOP_ND && targetFormat == TileOpFormat::TILEOP_FRACTAL_Z) {
        VecTile tmpVectile = relatedOp->GetTileShape().GetVecTile();
        tmpVectile.tile[0] = NUM16;
        tmpVectile.tile[1] = c0;
        tmpVectile.tile[3] *= tmpVectile.tile[2];
        tmpVectile.tile[2] = 1;
        TileShape::Current().SetVecTile(tmpVectile);
    } else if (srcFormat == TileOpFormat::TILEOP_ND && targetFormat == TileOpFormat::TILEOP_FRACTAL_Z_3D) {
        VecTile tmpVectile = relatedOp->GetTileShape().GetVecTile();
        tmpVectile.tile[0] = NUM16;
        tmpVectile.tile[1] = c0;
        tmpVectile.tile[4] *= tmpVectile.tile[3];
        tmpVectile.tile[3] = 1;
        TileShape::Current().SetVecTile(tmpVectile);
    } else if (srcFormat == TileOpFormat::TILEOP_ND && targetFormat == TileOpFormat::TILEOP_NDC1HWC0) {
        VecTile tmpVectile = relatedOp->GetTileShape().GetVecTile();
        tmpVectile.tile[0] = 1;
        tmpVectile.tile[1] *= NUM4;
        tmpVectile.tile[2] *= NUM2;
        tmpVectile.tile[3] *= NUM2;
        TileShape::Current().SetVecTile(tmpVectile);
    } else {
        TileShape::Current().SetVecTile(oriVectile);
    }
}

std::shared_ptr<LogicalTensor> InferTensorFormat::InsertTransDataOp(Function& function,
                                                                    const std::shared_ptr<LogicalTensor>& srcTensor,
                                                                    const std::shared_ptr<LogicalTensor>& fakeDstTensor,
                                                                    Operation* relatedOp, TileOpFormat targetFormat)
{
    int group_value = GetTransDataGroupValue(srcTensor, fakeDstTensor, relatedOp);

    Operation* convOp = FindRelatedConvOp(srcTensor, fakeDstTensor, targetFormat);
    VecTile savedVecTile = TileShape::Current().GetVecTile();
    bool vecTileInferred = false;
    if (convOp != nullptr) {
        auto vecTiles = Conv::InferConvVecTileShapes(*convOp, srcTensor->Datatype());
        VecTile inferredVecTile = Conv::SelectConvVecTile(vecTiles, targetFormat);
        if (!inferredVecTile.tile.empty()) {
            TileShape::Current().SetVecTile(inferredVecTile);
            vecTileInferred = true;
        }
    } else if (relatedOp != nullptr) {
        ApplyTransDataVecTile(srcTensor, targetFormat, relatedOp);
    }

    auto result = TransData(function, srcTensor, fakeDstTensor, targetFormat, group_value);
    result->GetRawTensor()->format = targetFormat;

    if (vecTileInferred) {
        TileShape::Current().SetVecTile(savedVecTile);
    }
    return result;
}

std::shared_ptr<LogicalTensor> InferTensorFormat::InsertFakeTransOp(Function& function,
                                                                    const std::shared_ptr<LogicalTensor>& srcTensor,
                                                                    TileOpFormat targetFormat, Operation* relatedOp)
{
    TileOpFormat srcFormat = srcTensor->Format();
    // FakeTrans 阶段 shape 暂保持源 shape，真正的 shape 变换在 Phase 3 物化时由 TransData() 完成
    auto outputTensor = irBuilder_.CreateTensorVar(function, srcTensor->Datatype(), srcTensor->GetShape(),
                                                   srcTensor->GetDynValidShape(), targetFormat);
    auto& fakeOp = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_FAKE_TRANS, {srcTensor}, {outputTensor});
    fakeOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(srcFormat));
    fakeOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR, static_cast<int64_t>(targetFormat));
    if (relatedOp != nullptr) {
        fakeOp.UpdateTileShape(relatedOp->GetTileShape());
    }
    outputTensor->GetRawTensor()->format = targetFormat;
    return outputTensor;
}

Status InferTensorFormat::EnsureTensorFormat(Function& function, std::shared_ptr<LogicalTensor>& tensor,
                                             Operation* relatedOp, TileOpFormat targetFormat)
{
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Null tensor when ensuring format %s for op [%s][%d].",
                          std::to_string(targetFormat).c_str(),
                          relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
                          relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
        return FAILED;
    }
    TileOpFormat current = tensor->Format();
    if (current == targetFormat || IsNdNzCompatibleFormat(current, targetFormat)) {
        return SUCCESS;
    }
    if (!IsSupportedTransData(current, targetFormat)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Unsupported format conversion: src=%s dst=%s for op [%s][%d].",
                          std::to_string(current).c_str(), std::to_string(targetFormat).c_str(),
                          relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
                          relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
        return FAILED;
    }
    std::shared_ptr<LogicalTensor> newTensor = InsertFakeTransOp(function, tensor, targetFormat, relatedOp);
    if (newTensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Insert TransData failed: tensor[%d] src=%s dst=%s for op [%s][%d].",
                          tensor->GetMagic(), std::to_string(current).c_str(), std::to_string(targetFormat).c_str(),
                          relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
                          relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "Inserted TransData: tensor[%d] (fmt=%s) -> tensor[%d] (fmt=%s) for op [%s][%d].",
                      tensor->GetMagic(), std::to_string(current).c_str(), newTensor->GetMagic(),
                      std::to_string(newTensor->Format()).c_str(),
                      relatedOp == nullptr ? "UNKNOWN" : relatedOp->GetOpcodeStr().c_str(),
                      relatedOp == nullptr ? -1 : relatedOp->GetOpMagic());
    tensor = newTensor;
    return SUCCESS;
}

Status InferTensorFormat::GetFakeTransFormat(const Operation& op, const std::string& attrName, TileOpFormat& format)
{
    if (!op.HasAttr(attrName)) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_FAKE_TRANS[%d] missing attribute [%s].", op.GetOpMagic(),
                          attrName.c_str());
        return FAILED;
    }
    int64_t value = op.GetIntAttribute(attrName);
    if (!IsValidTileOpFormat(value)) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_FAKE_TRANS[%d] attribute [%s] has invalid format value [%ld].",
                          op.GetOpMagic(), attrName.c_str(), value);
        return FAILED;
    }
    format = static_cast<TileOpFormat>(value);
    return SUCCESS;
}

// =============================================================================
// 输出 format 推导
// =============================================================================

void InferTensorFormat::DetermineOutputFormat(const Function& function, const Operation& op, const std::string& arch,
                                              std::unordered_map<int, bool>& visitedTensors,
                                              std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    Opcode opcode = op.GetOpcode();

    for (size_t i = 0; i < op.GetOOperands().size(); ++i) {
        auto output = op.GetOOperands()[i];
        TileOpFormat fmt;

        if (kPassThroughOps.count(opcode)) {
            fmt = op.GetIOperands()[0]->Format();
        } else if (opcode == Opcode::OP_FAKE_TRANS) {
            TileOpFormat fakeOutFmt;
            if (GetFakeTransFormat(op, FAKE_TRANS_OUT_FORMAT_ATTR, fakeOutFmt) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Failed to get FakeTrans output format for op [%s][%d].",
                                  op.GetOpcodeStr().c_str(), op.GetOpMagic());
                return;
            }
            fmt = fakeOutFmt;
        } else if (opcode == Opcode::OP_ASSEMBLE) {
            fmt = IsFunctionOutcast(function, output) ? TileOpFormat::TILEOP_ND : op.GetIOperands()[0]->Format();
        } else {
            fmt = GetOutputFormat(opcode, arch, i);
        }

        output->GetRawTensor()->format = fmt;

        int magic = output->GetMagic();
        if (visitedTensors.find(magic) == visitedTensors.end()) {
            visitedTensors[magic] = true;
            worklist.push(output);
        }
    }
}

void InferTensorFormat::EnqueueTensorIfNeeded(const std::shared_ptr<LogicalTensor>& tensor,
                                              std::unordered_map<int, bool>& visitedTensors,
                                              std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    if (tensor == nullptr) {
        return;
    }
    int magic = tensor->GetMagic();
    if (visitedTensors.find(magic) == visitedTensors.end()) {
        visitedTensors[magic] = true;
        worklist.push(tensor);
    }
}

void InferTensorFormat::EnqueueFunctionInputs(const Function& function, std::unordered_map<int, bool>& visitedTensors,
                                              std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    for (const auto& tensor : function.GetIncast()) {
        EnqueueTensorIfNeeded(tensor, visitedTensors, worklist);
    }
}

TileOpFormat InferTensorFormat::ResolveRequiredInputFormat(const Function& function, const Operation& consumer,
                                                           const std::shared_ptr<LogicalTensor>& tensor,
                                                           const std::string& arch, int inputPos,
                                                           std::unordered_set<int>& assembledOutputs)
{
    TileOpFormat required = GetRequiredInputFormat(consumer.GetOpcode(), arch, static_cast<size_t>(inputPos));
    if (consumer.GetOpcode() != Opcode::OP_ASSEMBLE || consumer.GetOOperands().empty()) {
        return required;
    }

    auto output = consumer.GetOOperands()[0];
    int outMagic = output->GetMagic();
    if (IsFunctionOutcast(function, output)) {
        return TileOpFormat::TILEOP_ND;
    }
    if (assembledOutputs.count(outMagic)) {
        return output->Format();
    }
    assembledOutputs.insert(outMagic);
    return tensor->Format();
}

Status InferTensorFormat::EnsureConsumerInputFormat(Function& function, Operation& consumer,
                                                    const std::shared_ptr<LogicalTensor>& tensor, TileOpFormat required)
{
    auto actualTensor = tensor;
    if (actualTensor->Format() == required) {
        return SUCCESS;
    }
    if (EnsureTensorFormat(function, actualTensor, &consumer, required) != SUCCESS) {
        return FAILED;
    }
    if (actualTensor->GetMagic() != tensor->GetMagic()) {
        consumer.ReplaceInput(actualTensor, tensor);
    }
    return SUCCESS;
}

void InferTensorFormat::MarkConsumerInputProcessed(const Function& function, const Operation& consumer,
                                                   const std::string& arch,
                                                   std::unordered_map<int, int>& processedInputs,
                                                   std::unordered_map<int, bool>& visitedTensors,
                                                   std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    int opMagic = consumer.GetOpMagic();
    processedInputs[opMagic]++;
    if (static_cast<size_t>(processedInputs[opMagic]) == consumer.GetInputOperandSize()) {
        DetermineOutputFormat(function, consumer, arch, visitedTensors, worklist);
    }
}

Status InferTensorFormat::ProcessConsumerFormat(Function& function, Operation* consumer,
                                                const std::shared_ptr<LogicalTensor>& tensor, const std::string& arch,
                                                std::unordered_map<int, int>& processedInputs,
                                                std::unordered_map<int, bool>& visitedTensors,
                                                std::unordered_set<int>& assembledOutputs,
                                                std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    if (consumer == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Null consumer for tensor [%d].", tensor->GetMagic());
        return FAILED;
    }
    if (consumer->IsDeleted()) {
        return SUCCESS;
    }

    int pos = FindInputPosition(*consumer, tensor);
    if (pos < 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "Tensor [%d] not in consumer [%d] inputs.", tensor->GetMagic(),
                          consumer->GetOpMagic());
        return FAILED;
    }
    if (consumer->GetOpcode() == Opcode::OP_FAKE_TRANS) {
        TileOpFormat fakeInFormat;
        TileOpFormat fakeOutFormat;
        if (GetFakeTransFormat(*consumer, FAKE_TRANS_IN_FORMAT_ATTR, fakeInFormat) != SUCCESS ||
            GetFakeTransFormat(*consumer, FAKE_TRANS_OUT_FORMAT_ATTR, fakeOutFormat) != SUCCESS) {
            return FAILED;
        }
        if (!IsSupportedTransData(fakeInFormat, fakeOutFormat)) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unsupported FakeTrans conversion: in=%s out=%s for op [%s][%d].",
                              std::to_string(fakeInFormat).c_str(), std::to_string(fakeOutFormat).c_str(),
                              consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic());
            return FAILED;
        }
        if (EnsureConsumerInputFormat(function, *consumer, tensor, fakeInFormat) != SUCCESS) {
            return FAILED;
        }
        MarkConsumerInputProcessed(function, *consumer, arch, processedInputs, visitedTensors, worklist);
        return SUCCESS;
    }

    TileOpFormat required = ResolveRequiredInputFormat(function, *consumer, tensor, arch, pos, assembledOutputs);
    if (EnsureConsumerInputFormat(function, *consumer, tensor, required) != SUCCESS) {
        return FAILED;
    }
    MarkConsumerInputProcessed(function, *consumer, arch, processedInputs, visitedTensors, worklist);
    return SUCCESS;
}

Status InferTensorFormat::ProcessTensorConsumers(Function& function, const std::shared_ptr<LogicalTensor>& tensor,
                                                 const std::string& arch, std::unordered_map<int, int>& processedInputs,
                                                 std::unordered_map<int, bool>& visitedTensors,
                                                 std::unordered_set<int>& assembledOutputs,
                                                 std::queue<std::shared_ptr<LogicalTensor>>& worklist)
{
    auto consumers = tensor->GetConsumers();
    for (auto* consumer : consumers) {
        if (ProcessConsumerFormat(function, consumer, tensor, arch, processedInputs, visitedTensors, assembledOutputs,
                                  worklist) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

// =============================================================================
// 主算法: BFS 从 incast 沿 consumer 链推导 format
// =============================================================================

Status InferTensorFormat::DeriveFormats(Function& function)
{
    std::string arch = NPUArchToString(Platform::Instance().GetSoc().GetNPUArch());

    std::queue<std::shared_ptr<LogicalTensor>> worklist;
    std::unordered_map<int, bool> visitedTensors;
    std::unordered_map<int, int> processedInputs;
    std::unordered_set<int> assembledOutputs; // 已被首个 assemble 初始化的输出 tensor

    EnqueueFunctionInputs(function, visitedTensors, worklist);

    for (auto& op : function.Operations()) {
        if (op.IsDeleted() || op.GetInputOperandSize() != 0) {
            continue;
        }
        DetermineOutputFormat(function, op, arch, visitedTensors, worklist);
    }

    APASS_LOG_DEBUG_F(Elements::Tensor, "Initial worklist size: %zu (incast + zero-input op outputs).",
                      worklist.size());

    while (!worklist.empty()) {
        auto tensor = worklist.front();
        worklist.pop();
        if (ProcessTensorConsumers(function, tensor, arch, processedInputs, visitedTensors, assembledOutputs,
                                   worklist) != SUCCESS) {
            return FAILED;
        }
    }

    return SUCCESS;
}

// =============================================================================
// Phase 2: 值编号+并查集消除冗余 FakeTrans
// =============================================================================

bool InferTensorFormat::IsFakeTransOutput(const std::shared_ptr<LogicalTensor>& tensor)
{
    for (auto* producer : tensor->GetProducers()) {
        if (producer != nullptr && !producer->IsDeleted() && producer->GetOpcode() == Opcode::OP_FAKE_TRANS) {
            return true;
        }
    }
    return false;
}

bool InferTensorFormat::IsValidFakeTransOp(const Operation& op)
{
    if (op.GetIOperands().size() != 1 || op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Invalid OP_FAKE_TRANS[%d]: expected 1 input and 1 output, got %zu input(s) and %zu output(s).",
            op.GetOpMagic(), op.GetIOperands().size(), op.GetOOperands().size());
        return false;
    }
    return true;
}

std::shared_ptr<LogicalTensor> InferTensorFormat::FindRoot(
    const std::shared_ptr<LogicalTensor>& tensor, std::unordered_map<int, std::shared_ptr<LogicalTensor>>& rootCache)
{
    int magic = tensor->GetMagic();
    auto it = rootCache.find(magic);
    if (it != rootCache.end()) {
        return it->second;
    }

    std::shared_ptr<LogicalTensor> fakeTransInput;
    bool isFakeTransOutput = false;
    for (auto* producer : tensor->GetProducers()) {
        if (producer != nullptr && !producer->IsDeleted() && producer->GetOpcode() == Opcode::OP_FAKE_TRANS) {
            if (!IsValidFakeTransOp(*producer)) {
                break; // Malformed FakeTrans — treat tensor as its own root
            }
            isFakeTransOutput = true;
            fakeTransInput = producer->GetIOperands()[0];
            break;
        }
    }

    if (!isFakeTransOutput) {
        rootCache[magic] = tensor;
        return tensor;
    }

    auto root = FindRoot(fakeTransInput, rootCache);
    rootCache[magic] = root;
    return root;
}

TileOpFormat InferTensorFormat::GetEffectiveFormat(const std::shared_ptr<LogicalTensor>& tensor)
{
    for (auto* producer : tensor->GetProducers()) {
        if (producer != nullptr && !producer->IsDeleted() && producer->GetOpcode() == Opcode::OP_FAKE_TRANS) {
            TileOpFormat outFmt;
            if (GetFakeTransFormat(*producer, FAKE_TRANS_OUT_FORMAT_ATTR, outFmt) == SUCCESS) {
                return outFmt;
            }
            break;
        }
    }
    return tensor->Format();
}

void InferTensorFormat::CollectSignatures(
    Function& function, std::unordered_map<int, std::shared_ptr<LogicalTensor>>& rootCache,
    std::map<Signature, std::vector<std::shared_ptr<LogicalTensor>>>& equivalenceClasses)
{
    // Collect FakeTrans output signatures
    for (auto& op : function.Operations()) {
        if (op.IsDeleted() || op.GetOpcode() != Opcode::OP_FAKE_TRANS) {
            continue;
        }
        if (!IsValidFakeTransOp(op)) {
            continue;
        }
        auto outputTensor = op.GetOOperands()[0];
        auto root = FindRoot(outputTensor, rootCache);
        TileOpFormat fmt = GetEffectiveFormat(outputTensor);
        Signature sig = {root->GetMagic(), static_cast<int>(fmt)};
        equivalenceClasses[sig].push_back(outputTensor);
    }

    // Collect non-FakeTrans output signatures (potential leaders)
    for (auto& op : function.Operations()) {
        if (op.IsDeleted()) {
            continue;
        }
        for (auto& output : op.GetOOperands()) {
            if (IsFakeTransOutput(output)) {
                continue;
            }
            Signature sig = {output->GetMagic(), static_cast<int>(output->Format())};
            equivalenceClasses[sig].push_back(output);
        }
    }

    // Collect incast tensors (function inputs)
    for (const auto& tensor : function.GetIncast()) {
        if (IsFakeTransOutput(tensor)) {
            continue;
        }
        Signature sig = {tensor->GetMagic(), static_cast<int>(tensor->Format())};
        equivalenceClasses[sig].push_back(tensor);
    }
}

bool InferTensorFormat::ReplaceRedundantTensors(
    std::map<Signature, std::vector<std::shared_ptr<LogicalTensor>>>& equivalenceClasses)
{
    constexpr size_t MIN_TENSORS_FOR_DEDUP = 2; // 等价类中至少需要 2 个 tensor 才存在冗余
    bool changed = false;
    for (auto& [sig, tensors] : equivalenceClasses) {
        (void)sig;
        if (tensors.size() < MIN_TENSORS_FOR_DEDUP) {
            continue;
        }
        // Pick leader: prefer non-FakeTrans output
        std::shared_ptr<LogicalTensor> leader = nullptr;
        for (auto& t : tensors) {
            if (!IsFakeTransOutput(t)) {
                leader = t;
                break;
            }
        }
        if (leader == nullptr) {
            leader = tensors[0];
        }
        // Replace others' consumers with leader
        for (auto& t : tensors) {
            if (t == leader) {
                continue;
            }
            auto consumers = t->GetConsumers(); // copy — GetConsumers returns std::set
            for (auto* consumer : consumers) {
                if (consumer != nullptr && !consumer->IsDeleted()) {
                    consumer->ReplaceInput(leader, t);
                    changed = true;
                }
            }
        }
    }
    return changed;
}

bool InferTensorFormat::DeleteDeadFakeTrans(Function& function)
{
    bool changed = false;
    for (auto& op : function.Operations()) {
        if (op.IsDeleted() || op.GetOpcode() != Opcode::OP_FAKE_TRANS) {
            continue;
        }
        if (!IsValidFakeTransOp(op)) {
            continue;
        }
        auto output = op.GetOOperands()[0];
        bool hasLiveConsumer = false;
        for (auto* consumer : output->GetConsumers()) {
            if (consumer != nullptr && !consumer->IsDeleted()) {
                hasLiveConsumer = true;
                break;
            }
        }
        if (!hasLiveConsumer) {
            op.SetAsDeleted();
            changed = true;
        }
    }
    return changed;
}

Status InferTensorFormat::EliminateRedundantFakeTrans(Function& function)
{
    do {
        std::unordered_map<int, std::shared_ptr<LogicalTensor>> rootCache;
        std::map<Signature, std::vector<std::shared_ptr<LogicalTensor>>> equivalenceClasses;

        CollectSignatures(function, rootCache, equivalenceClasses);
        bool changed = ReplaceRedundantTensors(equivalenceClasses);
        changed = DeleteDeadFakeTrans(function) || changed;

        if (!changed) {
            break;
        }
    } while (true);
    return SUCCESS;
}

// =============================================================================
// Phase 3: 物化剩余 FakeTrans 为真实 TransData
// =============================================================================

Status InferTensorFormat::MaterializeFakeTrans(Function& function)
{
    VecTile savedVecTile = TileShape::Current().GetVecTile();
    for (auto& op : function.Operations()) {
        if (op.IsDeleted() || op.GetOpcode() != Opcode::OP_FAKE_TRANS) {
            continue;
        }

        if (!IsValidFakeTransOp(op)) {
            TileShape::Current().SetVecTile(savedVecTile);
            return FAILED;
        }

        TileOpFormat inFmt;
        TileOpFormat outFmt;
        if (GetFakeTransFormat(op, FAKE_TRANS_IN_FORMAT_ATTR, inFmt) != SUCCESS ||
            GetFakeTransFormat(op, FAKE_TRANS_OUT_FORMAT_ATTR, outFmt) != SUCCESS) {
            TileShape::Current().SetVecTile(savedVecTile);
            return FAILED;
        }

        auto inputTensor = op.GetIOperands()[0];
        auto outputTensor = op.GetOOperands()[0];

        // Materialize: input format already matches IN_FORMAT (ensured by Phase 1)
        auto result = InsertTransDataOp(function, inputTensor, outputTensor, &op, outFmt);
        // Restore tileShape — ApplyTransDataVecTile modifies it globally
        TileShape::Current().SetVecTile(savedVecTile);
        if (result == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "MaterializeFakeTrans failed for OP_FAKE_TRANS[%d]: in=%s out=%s.",
                              op.GetOpMagic(), std::to_string(inFmt).c_str(), std::to_string(outFmt).c_str());
            return FAILED;
        }

        // Reconnect output's consumers to the materialized result
        auto consumers = outputTensor->GetConsumers(); // copy
        for (auto* consumer : consumers) {
            if (consumer != nullptr && !consumer->IsDeleted()) {
                consumer->ReplaceInput(result, outputTensor);
            }
        }

        op.SetAsDeleted();
    }
    return SUCCESS;
}

// =============================================================================
// Pass 入口
// =============================================================================

Status InferTensorFormat::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "Start InferTensorFormat for function [%s].", function.GetRawName().c_str());
    // 兜底告警前端输入中存在dynValidShape的tensor经过assemble后未手动view有效数据的情况，预警可能导致的精度问题。
    FunctionUtils::WarnAssembleDynValidShapeRisk(function);

    // Phase 1: BFS 推导格式，插入 FakeTrans 占位
    Status status = DeriveFormats(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "InferTensorFormat Phase 1 failed for function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }

    // Phase 2: 值编号+并查集消除冗余 FakeTrans
    status = EliminateRedundantFakeTrans(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "InferTensorFormat Phase 2 failed for function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }

    // Phase 3: 将剩余 FakeTrans 物化为真实 TransData
    status = MaterializeFakeTrans(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "InferTensorFormat Phase 3 failed for function [%s].",
                          function.GetRawName().c_str());
        return FAILED;
    }

    function.EraseOperations(true, false);

    APASS_LOG_INFO_F(Elements::Function, "End InferTensorFormat for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

} // namespace npu::tile_fwk
