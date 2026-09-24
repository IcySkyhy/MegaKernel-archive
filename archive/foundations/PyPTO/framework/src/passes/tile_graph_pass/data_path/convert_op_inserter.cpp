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
 * \file convert_op_inserter.cpp
 * \brief
 */
#include "convert_op_inserter.h"

#include <algorithm>
#include <unordered_set>

#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_log/pass_log.h"
#include "passes/tile_graph_pass/data_path/memory_path_utils.h"

#define MODULE_NAME "AssignMemoryType"

namespace npu {
namespace tile_fwk {

const std::unordered_set<DataType> kA2A3SupportedDtypes = {DT_INT4, DT_INT8, DT_UINT8, DT_FP16, DT_BF16, DT_INT16};
const std::unordered_set<DataType> kA5SupportedDtypes = {DT_INT4, DT_INT8, DT_UINT8, DT_FP16,
                                                         DT_BF16, DT_HF8,  DT_FP8,   DT_FP32};
const std::unordered_set<DataType> l0c2l1SupportedDtypes = {DT_FP16, DT_BF16};

const static std::unordered_map<NPUArch, std::unordered_set<DataType>> kArch2SupportedDtypes = {
    {NPUArch::DAV_1001, kA2A3SupportedDtypes},
    {NPUArch::DAV_2201, kA2A3SupportedDtypes},
    {NPUArch::DAV_3510, kA5SupportedDtypes},
    {NPUArch::DAV_UNKNOWN, kA2A3SupportedDtypes}};

void ConvertInserter::UpdateTensorTobeMap(const LogicalTensorPtr& tensor, Operation& operation, MemoryType t,
                                          const char* reason)
{
    bool hasReason = reason != nullptr && reason[0] != '\0';
    if (!tensor->HasConsumer(operation)) {
        if (hasReason) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "Operation %d is not a consumer of tensor %d; "
                              "Please make sure the operation is relative to the tensor to be mapped. reason: %s.",
                              operation.GetOpMagic(), tensor->GetMagic(), reason);
        } else {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "Operation %d is not a consumer of tensor %d; "
                              "Please make sure the operation is relative to the tensor to be mapped.",
                              operation.GetOpMagic(), tensor->GetMagic());
        }
        return;
    }
    int opMagic = operation.GetOpMagic();
    auto& tobeMap = tensorTobeMap[tensor];
    auto [opIt, inserted] = tobeMap.emplace(opMagic, std::make_pair(&operation, t));
    if (inserted) {
        return;
    }
    MemoryType& currentType = opIt->second.second;
    if (currentType == MemoryType::MEM_UNKNOWN && t != MemoryType::MEM_UNKNOWN) {
        currentType = t;
        return;
    }
    if (currentType == t) {
        return;
    }
    MemoryType oldType = currentType;
    currentType = t;
    if (hasReason) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Update tensor %d toBeMap(%s[%d]) from %s to %s, reason: %s.",
                          tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                          BriefMemoryTypeToString(oldType).c_str(), BriefMemoryTypeToString(t).c_str(), reason);
    } else {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Update tensor %d toBeMap(%s[%d]) from %s to %s,", tensor->GetMagic(),
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                          BriefMemoryTypeToString(oldType).c_str(), BriefMemoryTypeToString(t).c_str());
    }
}

std::map<MemoryType, std::set<Operation*, OpMagicComparator>> ConvertInserter::GetRequiredTobe(
    LogicalTensorPtr& tensor) const
{
    if (tensorTobeMap.count(tensor) == 0) {
        APASS_LOG_INFO_F(Elements::Tensor,
                         "Tensor %d has not been inserted yet; "
                         "Please make sure tensor in the tobe map.",
                         tensor->GetMagic());
        return {};
    }
    std::map<MemoryType, std::set<Operation*, OpMagicComparator>> result;
    for (const auto& item : tensorTobeMap.at(tensor)) {
        result[item.second.second].insert(item.second.first);
    }
    return result;
}

bool ConvertInserter::HasRequirement(const LogicalTensorPtr& tensor, const Operation& operation) const
{
    auto tensorIt = tensorTobeMap.find(tensor);
    if (tensorIt == tensorTobeMap.end()) {
        return false;
    }
    return tensorIt->second.count(operation.GetOpMagic()) > 0;
}

MemoryType ConvertInserter::GetRequirementOrUnknown(const LogicalTensorPtr& tensor, const Operation& operation) const
{
    auto tensorIt = tensorTobeMap.find(tensor);
    if (tensorIt == tensorTobeMap.end()) {
        return MemoryType::MEM_UNKNOWN;
    }
    auto opIt = tensorIt->second.find(operation.GetOpMagic());
    if (opIt == tensorIt->second.end()) {
        return MemoryType::MEM_UNKNOWN;
    }
    return opIt->second.second;
}

std::map<Operation*, MemoryType, OpMagicComparator> ConvertInserter::GetConsumerRequirements(
    const LogicalTensorPtr& tensor) const
{
    auto tensorIt = tensorTobeMap.find(tensor);
    if (tensorIt == tensorTobeMap.end()) {
        return {};
    }
    std::map<Operation*, MemoryType, OpMagicComparator> result;
    for (const auto& item : tensorIt->second) {
        result[item.second.first] = item.second.second;
    }
    return result;
}

std::set<MemoryType> ConvertInserter::GetKnownRequiredTypes(const LogicalTensorPtr& tensor) const
{
    auto tensorIt = tensorTobeMap.find(tensor);
    if (tensorIt == tensorTobeMap.end()) {
        return {};
    }
    std::set<MemoryType> result;
    for (const auto& item : tensorIt->second) {
        if (item.second.second != MemoryType::MEM_UNKNOWN) {
            result.insert(item.second.second);
        }
    }
    return result;
}

MemoryType ConvertInserter::TryGetUniqueKnownRequiredType(const LogicalTensorPtr& tensor) const
{
    auto knownTypes = GetKnownRequiredTypes(tensor);
    if (knownTypes.size() != 1) {
        return MemoryType::MEM_UNKNOWN;
    }
    return *knownTypes.begin();
}

void ConvertInserter::ClearPlanningState()
{
    conflictMap.clear();
    converts.clear();
    oldRawToNewRaw.clear();
}

std::map<MemoryType, std::set<Operation*, OpMagicComparator>> ConvertInserter::ReformMap(
    const std::map<int, std::pair<Operation*, MemoryType>>& oriMap) const
{
    std::map<MemoryType, std::set<Operation*, OpMagicComparator>> result;
    for (const auto& item : oriMap) {
        result[item.second.second].insert(item.second.first);
    }
    return result;
}

void ConvertInserter::FilterConflictTensor()
{
    for (const auto& pairLocal : tensorTobeMap) {
        auto tensorLocal = pairLocal.first;
        std::map<MemoryType, std::set<Operation*, OpMagicComparator>> tobeMap = ReformMap(pairLocal.second);
        if (tobeMap.size() == 1) {
            MemoryType requiredMemoryType = tobeMap.begin()->first;
            if (tensorLocal->GetMemoryTypeOriginal() == requiredMemoryType) {
                continue;
            }
        }
        conflictMap[tensorLocal->magic] = tobeMap;
    }
    APASS_LOG_INFO_F(Elements::Tensor, "--- ConflictMap size: %zu ---", conflictMap.size());
}

bool ConvertInserter::CrossCore(const MemoryType from, const MemoryType to) const
{
    std::vector<MemoryType> paths;
    Platform::Instance().GetDie().FindNearestPath(from, to, paths);

    return std::find(paths.begin(), paths.end(), MemoryType::MEM_DEVICE_DDR) != paths.end();
}

void ConvertInserter::UpdateConsumerAndReconnect(std::shared_ptr<LogicalTensor> oldTensor,
                                                 std::shared_ptr<LogicalTensor> newTensor, Operation* op) const
{
    newTensor->AddConsumer(op);
    auto updateViewOffset = [](std::shared_ptr<LogicalTensor> oldTensor1, std::shared_ptr<LogicalTensor> newTensor1,
                               Operation* op1) {
        auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op1->GetOpAttribute().get());
        if (viewOpAttribute != nullptr) {
            auto oldOffset = viewOpAttribute->GetFromOffset();
            if (oldTensor1->offset != newTensor1->offset) {
                for (size_t i = 0; i < oldOffset.size(); i++) {
                    oldOffset[i] -= oldTensor1->offset[i];
                }
            }
            viewOpAttribute->SetFromOffset(oldOffset, viewOpAttribute->GetFromDynOffset());
        }
    };
    for (size_t i = 0; i < op->iOperand.size(); ++i) {
        if ((op->iOperand[i]->magic == oldTensor->magic) &&
            (op->iOperand[i]->tensor->rawmagic == oldTensor->tensor->rawmagic)) {
            if (op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_SLICE) {
                updateViewOffset(oldTensor, newTensor, op);
            }
            op->ReplaceIOperand(i, newTensor);
        }
    }
}

Status ConvertInserter::RecordConflict(Function& function)
{
    oldRawToNewRaw.clear();
    converts.clear();
    std::unordered_set<int> visitedTensor;
    for (const auto& op : function.Operations()) {
        for (auto& iOperand : op.iOperand) {
            if (iOperand->GetProducers().size() >= 1) {
                continue;
            }
        }
        for (auto& oOperand : op.oOperand) {
            // step1:当tensor不在conflictMap中或者已经在visitedTensor中被处理过，可以跳过，否则需要处理内存冲突
            if (SkipOperand(oOperand, visitedTensor)) {
                continue;
            }
            // step2:解析冲突需求
            std::map<MemoryType, std::set<Operation*, OpMagicComparator>> tobeMap = conflictMap.at(oOperand->magic);
            for (const auto& item : tobeMap) {
                MemoryType requiredMemoryType = item.first;
                std::set<Operation*, OpMagicComparator> consumers = item.second;
                if (requiredMemoryType == oOperand->GetMemoryTypeOriginal()) {
                    continue;
                }

                // step3：处理特殊生产者消费者场景
                ProcessSpecialProducersOrConsumers(function, op, oOperand, consumers, requiredMemoryType);
                if (requiredMemoryType == oOperand->GetMemoryTypeOriginal()) {
                    continue;
                }

                // step4:构造转换路径
                std::vector<MemoryType> paths;
                Status status = ProcessConvertPath(op, oOperand, requiredMemoryType, paths);
                if (status != SUCCESS) {
                    return status;
                }

                // step5：对每个消费者插入的Convert Op并更新图链接
                InsertConvertOpForEachConsumer(function, op, oOperand, consumers, paths);

                // step6：标记已处理
                visitedTensor.insert(oOperand->magic);
            }
        }
    }
    return SUCCESS;
}

void ConvertInserter::InsertConvertOpForEachConsumer(Function& function, const Operation& op,
                                                     const std::shared_ptr<LogicalTensor>& oOperand,
                                                     std::set<Operation*, OpMagicComparator>& consumers,
                                                     std::vector<MemoryType>& paths)
{
    for (auto consumer : consumers) {
        auto output = RecordInsertConvertOp(oOperand, paths, function, op);
        if (consumer->BelongTo() == &function) {
            UpdateConsumerAndReconnect(oOperand, output, consumer);
        }
    }
}

// 特殊场景处理：生成者均为Contract或者消费者均为Slice/Contract，且mem路径中经过DDR
void ConvertInserter::ProcessSpecialProducersOrConsumers(Function& function, const Operation& op,
                                                         const std::shared_ptr<LogicalTensor>& oOperand,
                                                         std::set<Operation*, OpMagicComparator>& consumers,
                                                         MemoryType& requiredMemoryType)
{
    // case1:当tensor的生产者都是contract，并且tensor的mem路径需要经过DDR，则将tensor的ori刷成DDR
    APASS_LOG_DEBUG_F(Elements::Operation, "Operation %s[%d] has output %d original and requirement conflict.",
                      op.GetOpcodeStr().c_str(), op.GetOpMagic(), oOperand->magic);
    const auto& items = tensorTobeMap.at(oOperand);
    bool crossCore = std::all_of(items.begin(), items.end(), [this, &oOperand](const auto& item) {
        return CrossCore(oOperand->GetMemoryTypeOriginal(), item.second.second);
    });
    bool producedByContract = IsAllProducerContract(oOperand);
    if (producedByContract && crossCore) {
        oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    }
    // case2:当tensor的消费者都是slice或者contract，并且tensor的mem路径需要经过DDR时，将需求降级为DDR
    bool canSetBoth = IsAllMoveConsumersValid(function, consumers);
    if (canSetBoth && crossCore) {
        requiredMemoryType = MEM_DEVICE_DDR;
    }
}

bool ConvertInserter::IsNotValidDataType(const std::shared_ptr<LogicalTensor>& firstCVOutput) const
{
    // 1. 获取当前NPU架构
    const NPUArch currentArch = Platform::Instance().GetSoc().GetNPUArch();

    // 2. 查找当前架构对应的支持类型集合（容错：找不到则用DAV_UNKNOWN兜底）
    auto archIter = kArch2SupportedDtypes.find(currentArch);
    if (archIter == kArch2SupportedDtypes.end()) {
        archIter = kArch2SupportedDtypes.find(NPUArch::DAV_UNKNOWN);
    }
    const auto& supportedDtypes = archIter->second;

    // 3. 判断当前张量类型是否不在支持列表中（不在则返回true，表示无效）
    const DataType tensorDtype = firstCVOutput->Datatype();
    return supportedDtypes.find(tensorDtype) == supportedDtypes.end();
}

// Tensor必须是BF16或FP16，同时矩阵必须是第一轴（外轴）16元素对齐，第二轴（内轴）32B对齐
bool ConvertInserter::FitL0C2L1(const LogicalTensorPtr& tensor)
{
    auto shape = tensor->GetShape();
    if (shape.size() != MATMUL_DIM_NUM) {
        return false;
    }
    auto dim2Size = shape[1] * BytesOf(tensor->Datatype());
    return (l0c2l1SupportedDtypes.find(tensor->Datatype()) != l0c2l1SupportedDtypes.end()) &&
           (shape[0] % L0C2L1_DIM1_SHAPE_RESTICT == 0) && (dim2Size % L0C2L1_DIM2_BYTE_RESTICT == 0);
}

// 规避条件检测，检查op是否满足输入无表达式validShape。
// 规避问题： L0C2L1的输入存在validShape时，即便输出同样存在validShape也会导致精度问题，此场景暂时走DDR规避。
bool ConvertInserter::FitL0C2L1(const Operation& op)
{
    auto isSmallToLarge = [&op]() {
        const auto& inShape = op.iOperand.front()->GetShape();
        const auto& outShape = op.oOperand.front()->GetShape();
        if (inShape.size() != outShape.size() || inShape.empty()) {
            return false;
        }
        for (size_t i = 0; i < inShape.size(); ++i) {
            if (!(inShape[i] <= outShape[i])) {
                return false;
            }
        }
        return true;
    };
    // pto-isa 不支持 TLOAD NZ 格式的 stride 跳变，在小搬大场景下做规避
    if (isSmallToLarge() && op.iOperand.front()->GetShape()[0] != op.oOperand.front()->GetShape()[0]) {
        return false;
    }
    for (const auto& input : op.GetIOperands()) {
        const auto& dynValidShape = input->GetDynValidShape();
        for (const auto& dim : dynValidShape) {
            if (!dim.IsImmediate()) {
                return false;
            }
        }
    }
    auto in = op.iOperand.front();
    return FitL0C2L1(in);
}

bool ConvertInserter::FitUB2L1(const LogicalTensorPtr& tensor) const
{
    auto shape = tensor->GetShape();
    if (shape.size() != MATMUL_DIM_NUM) {
        return false;
    }
    return true;
}

bool ConvertInserter::HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor,
                                                              MemoryType targetType) const
{
    if (tensor == nullptr || tensor->GetConsumers().size() <= 1) {
        return false;
    }
    auto tensorIt = tensorTobeMap.find(tensor);
    if (tensorIt == tensorTobeMap.end()) {
        return false;
    }
    return std::any_of(tensorIt->second.begin(), tensorIt->second.end(), [this, targetType](const auto& item) {
        auto resolveOutputRequirement = [this](const LogicalTensorPtr& output) {
            return TryGetUniqueKnownRequiredType(output);
        };
        MemoryType requirement = MemoryPathUtils::ResolveEffectiveConsumerRequirement(
            item.second.first, item.second.second, targetType, resolveOutputRequirement);
        return MemoryPathUtils::IsDifferentKnownRequirement(requirement, targetType);
    });
}

Status ConvertInserter::ProcessConvertPath(const Operation& op, const std::shared_ptr<LogicalTensor>& oOperand,
                                           MemoryType requiredMemoryType, std::vector<MemoryType>& paths)
{
    auto currTensorMemOri = oOperand->GetMemoryTypeOriginal();
    bool useDdrForSpecialPath = MemoryPathUtils::IsSpecialDirectMemoryPath(currTensorMemOri, requiredMemoryType) &&
                                HasParallelDifferentConsumerRequirement(oOperand, requiredMemoryType);
    if (useDdrForSpecialPath) {
        paths = {currTensorMemOri, MemoryType::MEM_DEVICE_DDR, requiredMemoryType};
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Use DDR fallback path for tensor %d because %s -> %s has parallel different requirements.",
                          oOperand->magic, BriefMemoryTypeToString(currTensorMemOri).c_str(),
                          BriefMemoryTypeToString(requiredMemoryType).c_str());
        return SUCCESS;
    }
    if (currTensorMemOri == MemoryType::MEM_L0C && requiredMemoryType == MemoryType::MEM_L1) {
        // 特殊处理L0C2L1：针对不支持的数据类型场景路径中插入DDR
        bool needDDRTrans = IsNotValidDataType(oOperand) || !FitL0C2L1(op);
        if (needDDRTrans) {
            paths = {currTensorMemOri, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1};
        } else {
            paths = {currTensorMemOri, MemoryType::MEM_L1};
        }
    } else {
        // 常规场景：查platform硬件配置获取path处理
        Status status = ConstructPath(oOperand->GetMemoryTypeOriginal(), requiredMemoryType, paths, oOperand, op);
        if (status != SUCCESS) {
            return status;
        }
    }
    return SUCCESS;
}

Status ConvertInserter::ConstructPath(MemoryType from, MemoryType to, std::vector<MemoryType>& paths,
                                      const std::shared_ptr<LogicalTensor>& oOperand, const Operation& op) const
{
    Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (paths.empty()) {
        // path为空的两种场景:1、from和to内存类型一致；2、from和to不一致，且未找到数据通路。这里处理场景2，报错退出
        APASS_LOG_ERROR_F(Elements::Operation,
                          "No memory path found from %s to %s for tensor %d in operation %s[%d]. %s",
                          BriefMemoryTypeToString(from).c_str(), BriefMemoryTypeToString(to).c_str(), oOperand->magic,
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

bool ConvertInserter::SkipOperand(const std::shared_ptr<LogicalTensor>& oOperand,
                                  const std::unordered_set<int>& visitedTensor) const
{
    return ((conflictMap.find(oOperand->magic) == conflictMap.end()) ||
            (visitedTensor.find(oOperand->magic) != visitedTensor.end()));
}

bool ConvertInserter::IsAllProducerContract(const std::shared_ptr<LogicalTensor>& oOperand) const
{
    auto producers = oOperand->GetProducers();
    return std::all_of(producers.begin(), producers.end(),
                       [](const Operation* producerOp) { return producerOp->GetOpcode() == Opcode::OP_CONTRACT; });
}

bool ConvertInserter::IsAllMoveConsumersValid(Function& function,
                                              const std::set<Operation*, OpMagicComparator>& consumers) const
{
    for (const auto consumer : consumers) {
        if (consumer->GetOpcode() != Opcode::OP_SLICE && consumer->GetOpcode() != Opcode::OP_CONTRACT) {
            return false;
        }
        if (consumer->GetOpcode() == Opcode::OP_CONTRACT &&
            FunctionUtils::GetNodeType(*(consumer->GetOOperands().front()), function) != NodeType::OUTCAST) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<LogicalTensor> ConvertInserter::RecordInsertConvertOp(const std::shared_ptr<LogicalTensor>& oOperand,
                                                                      const std::vector<MemoryType>& paths,
                                                                      Function& function, const Operation& op)
{
    (void)function;
    std::shared_ptr<LogicalTensor> input = oOperand;
    for (size_t i = 0; i < paths.size() - 1; ++i) {
        std::shared_ptr<LogicalTensor> output = CreateTensorLikeForConvert(input, paths[i + 1]);
        converts.emplace_back(ConvertOpInfo{paths[i], paths[i + 1], input, output});
        APASS_LOG_DEBUG_F(Elements::Tensor, "%s[%d] --> tensor[%d](%s) --> Convert --> %s.", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic(), input->magic,
                          BriefMemoryTypeToString(input->GetMemoryTypeOriginal()).c_str(),
                          BriefMemoryTypeToString(output->GetMemoryTypeOriginal()).c_str());
        APASS_LOG_DEBUG_F(Elements::Tensor, "--- %s --> %s ---", BriefMemoryTypeToString(paths[i]).c_str(),
                          BriefMemoryTypeToString(paths[i + 1]).c_str());
        input = output;
    }
    return input;
}

std::shared_ptr<LogicalTensor> ConvertInserter::CreateTensorLikeForConvert(const std::shared_ptr<LogicalTensor>& input,
                                                                           MemoryType outputMemoryType) const
{
    IRBuilder builder;
    std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(input->Datatype(), input->GetShape(),
                                                                          input->Format());
    std::vector<int64_t> newoffset(input->offset.size(), 0);
    std::shared_ptr<LogicalTensor> output = builder.CreateTensorVar(newRawTensor, newoffset, input->shape,
                                                                    std::vector<SymbolicScalar>{});
    output->SetMemoryTypeOriginal(outputMemoryType);
    GraphUtils::CopyDynStatus(output, input);
    return output;
}

void ConvertInserter::GraphReconnect(const std::shared_ptr<LogicalTensor>& oOperand,
                                     std::shared_ptr<LogicalTensor> output,
                                     const std::set<Operation*, OpMagicComparator>& consumers, Function& function) const
{
    for (const auto& consumer : consumers) {
        if (consumer->BelongTo() == &function) {
            UpdateConsumerAndReconnect(oOperand, output, consumer);
        }
    }
}

bool ConvertInserter::CreateMoveOpForConvert(Operation& op)
{
    auto convertOpAttribute = dynamic_cast<ConvertOpAttribute*>(op.GetOpAttribute().get());
    auto [from, to] = convertOpAttribute->GetConvertPath();

    if (from == MemoryType::MEM_DEVICE_DDR) {
        op.SetOpCode(Opcode::OP_SLICE); // 将convert改为搬入操作，后续GenerateMoveOp Pass会转化为copyin
        op.SetOpAttribute(BuildSliceAttrForConvert(op, to));
        auto childOp = *op.oOperand.front()->GetConsumers().begin();
        op.UpdateSubgraphID(childOp->GetSubgraphID());
        op.SetScopeInfo(childOp->GetScopeInfo());
        return true;
    }

    if (to == MemoryType::MEM_DEVICE_DDR) {
        op.SetOpCode(Opcode::OP_CONTRACT); // 将convert改为搬出操作，后续GenerateMoveOp Pass会转化为copyout
        op.SetOpAttribute(BuildContractAttrForConvert(op, from));
        auto parentOp = *op.iOperand.front()->GetProducers().begin();
        op.UpdateSubgraphID(parentOp->GetSubgraphID());
        op.SetScopeInfo(parentOp->GetScopeInfo());
        return true;
    }
    return false;
}

void ConvertInserter::InsertConvertOps(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "--- Need to insert %zu convert operations ---", converts.size());
    for (const auto& c : converts) {
        IRBuilder builder;
        auto& convertOp = builder.CreateTensorOpStmt(function, Opcode::OP_CONVERT, {c.input}, {c.output});
        convertOp.SetOpAttribute(std::make_shared<ConvertOpAttribute>(c.from, c.to));
        if (!CreateMoveOpForConvert(convertOp)) {
            auto producerScopeInfo = (*(c.input->GetProducers().begin()))->GetScopeInfo();
            convertOp.SetScopeInfo(producerScopeInfo); // convert 是拷贝出操作，和producer一个子图
        }
    }
}

std::shared_ptr<ViewOpAttribute> ConvertInserter::BuildSliceAttrForConvert(const Operation& op, MemoryType to) const
{
    auto input = op.iOperand.front();
    return std::make_shared<ViewOpAttribute>(input->GetOffset(), to, input->GetDynOffset(), input->GetDynValidShape());
}

std::shared_ptr<AssembleOpAttribute> ConvertInserter::BuildContractAttrForConvert(const Operation& op,
                                                                                  MemoryType from) const
{
    auto input = op.iOperand.front();
    auto output = op.oOperand.front();
    return std::make_shared<AssembleOpAttribute>(from, output->GetOffset(), output->GetDynOffset(),
                                                 input->GetDynValidShape());
}

// 对外总接口
Status ConvertInserter::DoInsertion(Function& function)
{
    ClearPlanningState();
    FilterConflictTensor();
    Status status = RecordConflict(function);
    if (status != SUCCESS) {
        return status;
    }
    InsertConvertOps(function);
    APASS_LOG_INFO_F(Elements::Function, "After Insert Convert, total op Num: %zu.", function.GetOperationSize());
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
