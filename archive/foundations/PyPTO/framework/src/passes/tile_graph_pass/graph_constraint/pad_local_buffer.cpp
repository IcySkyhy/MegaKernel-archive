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
 * \file pad_local_buffer.cpp
 * \brief
 */

#include "pad_local_buffer.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/alignment_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include <numeric>

#define MODULE_NAME "PadLocalBuffer"

namespace npu::tile_fwk {
constexpr size_t MATMUL_MIN_SHAPE_SIZE = 2;
constexpr int64_t CUBE_PAD_VALUE = 16;
constexpr int64_t CUBE_PAD_B8_VALUE = 32;
constexpr int64_t CUBE_PAD_B4_VALUE = 64;
constexpr int64_t MX_K_AXIS_PAD_VALUE = 64;
constexpr int64_t BT_PAD_BASE = 64;
constexpr int64_t mxHighAxis = 0;
constexpr int64_t mxLowAxis = 1;
const std::unordered_set<DataType> b8DataSupport = {DataType::DT_INT8, DataType::DT_FP8E5M2, DataType::DT_FP8E4M3,
                                                    DataType::DT_HF8};
const std::unordered_set<DataType> b4DataSupport = {DataType::DT_FP4_E2M1X2, DataType::DT_FP4_E1M2X2,
                                                    DataType::DT_FP4_E2M1, DataType::DT_FP4_E1M2};
// combine_axis
const int64_t BRCB_SECOND_LAST_BASE = 8;
const size_t LAST_SECOND_AXIS = 2;
const std::unordered_set<OpCalcType> ELEMENTWISE_LIKE_TYPES{OpCalcType::CAST, OpCalcType::ELMWISE, OpCalcType::MOVE_IN,
                                                            OpCalcType::MOVE_OUT};
// OP_VIEW_TYPE 当前不在白名单内，所在通路不支持合轴，合轴模式下不会对其进行倒数第二轴 pad
const std::unordered_set<Opcode> shapeTransformOps{Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_RESHAPE};
// 设置原始 rawshape（替代 RawTensor::oriRawshape = rawshape）
// 原始代码中 oriRawshape 是每次进入函数时都设置为当前 rawshape 值
Shape& PadLocalBuffer::SetOriRawshape(LogicalTensorPtr& in)
{
    int rawmagic = in->tensor->rawmagic;
    oriRawshapeMap_[rawmagic] = in->tensor->rawshape;
    return oriRawshapeMap_[rawmagic];
}

// 获取已保存的原始 rawshape，如果不存在则返回当前 rawshape
Shape& PadLocalBuffer::GetOriRawshape(LogicalTensorPtr& in)
{
    int rawmagic = in->tensor->rawmagic;
    if (oriRawshapeMap_.find(rawmagic) == oriRawshapeMap_.end()) {
        APASS_LOG_WARN_F(Elements::Tensor, "oriRawshape not set for tensor %d, fallback to current rawshape",
                         in->tensor->rawmagic);
        oriRawshapeMap_[rawmagic] = in->tensor->rawshape;
    }
    return oriRawshapeMap_[rawmagic];
}

bool PadLocalBuffer::IsInputDataType(const Operation& op, const LogicalTensorPtr& in,
                                     const std::unordered_set<DataType>& targetTypes) const
{
    static const std::unordered_set<Opcode> cubeOps = {Opcode::OP_A_MUL_B, Opcode::OP_AT_MUL_B, Opcode::OP_A_MUL_BT,
                                                       Opcode::OP_AT_MUL_BT, Opcode::OP_A_MULACC_B};

    if (in == nullptr || in->tensor == nullptr) {
        return false;
    }

    auto inputMemType = in->GetMemoryTypeOriginal();
    if (inputMemType == MemoryType::MEM_L0C) {
        // L0C上只需要对齐到(16, 16)
        return false;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Matmul Op %d is %s\n", op.opmagic, op.GetOpcodeStr().c_str());
    APASS_LOG_DEBUG_F(Elements::Tensor, "####### %d data type is %s\n", in->magic,
                      DataType2VectorRegStr(in->tensor->GetDataType()).c_str());

    bool matmulOp = cubeOps.count(op.GetOpcode()) != 0;
    bool opsInputDtype = false;
    if (op.GetIOperands().size() > 0 && op.GetIOperands()[0] != nullptr && op.GetIOperands()[0]->tensor != nullptr) {
        opsInputDtype = targetTypes.find(op.GetIOperands()[0]->tensor->GetDataType()) != targetTypes.end();
    }

    if (targetTypes.find(in->tensor->GetDataType()) != targetTypes.end() || (matmulOp && opsInputDtype)) {
        // 检查op的输入数据类型是不是int8类型或者in的数据类型是否为int8
        // 包括matmul系列和GM->L1->L0系列
        return true;
    }

    if (op.GetOpcode() == Opcode::OP_COPY_OUT || op.GetOpcode() == Opcode::OP_L0C_TO_L1 ||
        op.GetOpcode() == Opcode::OP_L0C_COPY_UB || op.GetOpcode() == Opcode::OP_UB_COPY_L1) {
        Operation* inProducerPtr = *in->GetProducers().begin();
        if (inProducerPtr != nullptr && inProducerPtr->GetIOperands().size() != 0 &&
            inProducerPtr->GetIOperands()[0] != nullptr && inProducerPtr->GetIOperands()[0]->tensor != nullptr) {
            // 检查in的前置op节点的输入是否为int8。
            // iOperands (dtype:int8) --> A_MULACC_B --> in (dtype:fp16/int32), iOperands (dtype:fp16/int32) -->
            // COPY_OUT
            return targetTypes.find(inProducerPtr->GetIOperands()[0]->tensor->GetDataType()) != targetTypes.end();
        }
    }
    return false;
}

void PadLocalBuffer::PadMatmulL1ConvertScene(Operation& op, LogicalTensorPtr& in, size_t lowIndex)
{
    const auto& producers = in->GetProducers();
    const auto& consumers = in->GetConsumers();
    auto bytes = BytesOf(in->Datatype());
    auto& padShape = in->tensor->rawshape;
    Shape& oriRawshape = GetOriRawshape(in);                        // 获取已保存的 oriRawshape
    if ((*producers.begin())->GetOpcode() == Opcode::OP_L1_TO_BT) { // Opcode::OP_L1_TO_BT input 和 output shape 一致
        const auto& preInput = (*producers.begin())->GetIOperands().front();
        padShape = preInput->tensor->rawshape;
        return;
    }
    if ((*consumers.begin())->GetOpcode() == Opcode::OP_L1_TO_BT) { // Opcode::OP_L1_TO_BT
        if (bytes == 0 || BT_PAD_BASE % bytes != 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Matmul Op %d %s input %d type is not valid.", op.opmagic,
                              op.GetOpcodeStr().c_str(), in->magic);
            return;
        }
        padShape[lowIndex] = AlignmentUtils::Pad(oriRawshape[lowIndex], BT_PAD_BASE / bytes);
    } else if ((*producers.begin())->GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE ||
               (*consumers.begin())->GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE) { // Opcode::OP_L1_TO_FIX_QUANT_PRE
        padShape[lowIndex] = AlignmentUtils::Pad(oriRawshape[lowIndex], CUBE_PAD_VALUE);
    }
}

void PadLocalBuffer::PadForMatMulMX(LogicalTensorPtr& in, const int64_t& axisNum)
{
    Shape& oriRawshape = SetOriRawshape(in); // 先设置，再使用
    in->tensor->rawshape[axisNum] = AlignmentUtils::Pad(oriRawshape[axisNum], CUBE_PAD_B8_VALUE);
}

void PadLocalBuffer::TryPadMatmulIsMXScene(Operation& op, LogicalTensorPtr& in)
{
    if (in == nullptr || in->tensor == nullptr || in->tensor->rawshape.size() < MATMUL_MIN_SHAPE_SIZE) {
        return;
    }
    const auto& producers = in->GetProducers();
    if (producers.empty() || *producers.begin() == nullptr ||
        (!op.GetBoolAttribute("op_attr_is_mx") && !(*producers.begin())->GetBoolAttribute("op_attr_is_mx"))) {
        return;
    }
    size_t highIndex = in->tensor->rawshape.size() - 2;
    size_t lowIndex = in->tensor->rawshape.size() - 1;
    auto memType = in->GetMemoryTypeOriginal();
    auto opcode = op.GetOpcode();
    size_t kAxis = [&]() -> size_t {
        switch (memType) {
            case MemoryType::MEM_L0A:
                return lowIndex;
            case MemoryType::MEM_L0B:
                return highIndex;
            case MemoryType::MEM_L1:
                switch (opcode) {
                    case Opcode::OP_L1_TO_L0A:
                    case Opcode::OP_L1_TO_L0_BT:
                        return lowIndex;
                    case Opcode::OP_L1_TO_L0_AT:
                    case Opcode::OP_L1_TO_L0B:
                        return highIndex;
                    default:
                        return -1; // invalid case
                }
            default:
                return -1; // invalid case
        }
    }();
    if (kAxis == static_cast<size_t>(-1)) {
        return;
    }
    in->tensor->rawshape[kAxis] = AlignmentUtils::Pad(in->tensor->rawshape[kAxis], MX_K_AXIS_PAD_VALUE);
}

bool PadLocalBuffer::TryPadMatmulMXScene(Operation& op, LogicalTensorPtr& in)
{
    const auto& producers = in->GetProducers();
    if (producers.empty()) {
        return false;
    }
    auto producerOpcode = (*producers.begin())->GetOpcode();
    if (op.GetOpcode() == Opcode::OP_L1_TO_L0A_SCALE || producerOpcode == Opcode::OP_L1_TO_L0A_SCALE) {
        PadForMatMulMX(in, mxHighAxis);
        return true;
    }
    if (op.GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE || producerOpcode == Opcode::OP_L1_TO_L0B_SCALE) {
        PadForMatMulMX(in, mxLowAxis);
        return true;
    }
    return false;
}

int64_t PadLocalBuffer::GetMatmulPaddingValue(Operation& op, LogicalTensorPtr& in) const
{
    if (IsInputDataType(op, in, b8DataSupport)) {
        return CUBE_PAD_B8_VALUE;
    }
    if (IsInputDataType(op, in, b4DataSupport)) {
        return CUBE_PAD_B4_VALUE;
    }
    return CUBE_PAD_VALUE;
}

void PadLocalBuffer::PadMatmulHighLow(LogicalTensorPtr& in, size_t highIndex, size_t lowIndex, int64_t padValue)
{
    Shape& oriRawshape = GetOriRawshape(in); // 获取已保存的 oriRawshape
    in->tensor->rawshape[highIndex] = AlignmentUtils::Pad(oriRawshape[highIndex], padValue);
    in->tensor->rawshape[lowIndex] = AlignmentUtils::Pad(oriRawshape[lowIndex], padValue);
}

bool PadLocalBuffer::IsGemvL1ATensor(const LogicalTensorPtr& in) const
{
    if (in == nullptr || in->GetMemoryTypeOriginal() != MemoryType::MEM_L1) {
        return false;
    }
    // isGemv 由 LinkTensorA 写入 DDR→L1 VIEW，经 MERGE_VIEW_ASSEMBLE 白名单继承到 COPY_IN，直接读取
    const auto& producers = in->GetProducers();
    if (producers.empty() || *producers.begin() == nullptr) {
        return false;
    }
    auto producer = *producers.begin();
    int64_t isGemv = 0;
    if (!(producer->GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv) && isGemv != 0)) {
        return false;
    }
    // 排除 transA：其 L1 A 为 {K,1}，由 OP_L1_TO_L0_AT 消费，去 pad/512B 对齐会破坏布局
    const auto& consumers = in->GetConsumers();
    if (consumers.empty() || *consumers.begin() == nullptr ||
        (*consumers.begin())->GetOpcode() != Opcode::OP_L1_TO_L0A) {
        return false;
    }
    return true;
}

bool PadLocalBuffer::IsGemvL0ATensor(const LogicalTensorPtr& in) const
{
    if (in == nullptr || in->GetMemoryTypeOriginal() != MemoryType::MEM_L0A) {
        return false;
    }
    const auto& producers = in->GetProducers();
    if (producers.empty() || *producers.begin() == nullptr) {
        return false;
    }
    auto producer = *producers.begin();
    int64_t isGemv = 0;
    return producer->GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv) && isGemv != 0;
}

void PadLocalBuffer::PadMatmulGemvKAlign(LogicalTensorPtr& in, size_t highIndex, size_t lowIndex)
{
    // GEMV 走 pto-isa TExtractToAVector：L1 源按 32B block 寻址，L0A 目标按 512B fractal block 组织
    Shape& oriRawshape = GetOriRawshape(in); // 获取已保存的 oriRawshape
    auto bytes = BytesOf(in->Datatype());
    if (in->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
        constexpr int64_t kAlignFractalByteSize = 512; // fractal 粒度（CUBE_BLOCK_SIZE）
        int64_t kAlignFractal = (bytes == 0) ? CUBE_PAD_VALUE : kAlignFractalByteSize / bytes;
        in->tensor->rawshape[lowIndex] = AlignmentUtils::Pad(oriRawshape[lowIndex], kAlignFractal);
    } else {
        constexpr int64_t kAlignBlockByteSize = 32; // TExtractToAVector 的 32B block 粒度
        int64_t kAlign32 = (bytes == 0) ? CUBE_PAD_VALUE : kAlignBlockByteSize / bytes;
        in->tensor->rawshape[lowIndex] = AlignmentUtils::Pad(oriRawshape[lowIndex], kAlign32);
    }
    // GEMV A 的 M 轴不打 pad：L1/L0A 均保持 1
    in->tensor->rawshape[highIndex] = oriRawshape[highIndex];
}

void PadLocalBuffer::PadMatmul(Operation& op, LogicalTensorPtr& in)
{
    if (in == nullptr || in->tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "logical tensor pointer is null.");
        return;
    }
    if (in->shape.size() < MATMUL_MIN_SHAPE_SIZE) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Matmul Op %d %s input %d shape size is less than 2; Please check the input size. %s",
                          op.opmagic, op.GetOpcodeStr().c_str(), in->magic, GetFormatBacktrace(op).c_str());
        return;
    }
    auto highIndex = in->shape.size() - 2; // matmul高轴
    auto lowIndex = in->shape.size() - 1;  // matmul低轴
    const auto& producers = in->GetProducers();
    const auto& consumers = in->GetConsumers();
    const bool isL1ConvertScene = !producers.empty() && !consumers.empty() && *producers.begin() != nullptr &&
                                  *consumers.begin() != nullptr &&
                                  ((*producers.begin())->GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE ||
                                   (*producers.begin())->GetOpcode() == Opcode::OP_L1_TO_BT ||
                                   (*consumers.begin())->GetOpcode() == Opcode::OP_L1_TO_BT ||
                                   (*consumers.begin())->GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE);
    const bool isUB2L1Scene = !consumers.empty() && *consumers.begin() != nullptr &&
                              (*consumers.begin())->GetOpcode() == Opcode::OP_UB_COPY_L1;
    // GEMV A 来自 UB（isGemv 由 ProcessUB2L1 设置）：UB_COPY_L1 输入按 K 对齐、M 不 pad
    int64_t isGemvUbCopyL1Val = 0;
    const bool isGemvUbCopyL1 = isUB2L1Scene &&
                                (*consumers.begin())->GetAttr<int64_t>(OpAttributeKey::isGemv, isGemvUbCopyL1Val) &&
                                isGemvUbCopyL1Val != 0;
    /*
    首先，可以通过in的数据类型是否为int8来判断是否要做32B对齐。
    再者，存在两种情况
    第一种：in (dtype:fp16/int32) -> iOperands (dtype:int8) -> Matmul系列(A_MUL_B, AT_MUL_B, A_MUL_BT, AT_MUL_BT)
    这种情况需要通过op.GetIOperands来判断输入是否为int8。
    第二种：iOperands (dtype:int8) -> Matmul系列(A_MUL_B, AT_MUL_B, A_MUL_BT, AT_MUL_BT, A_MULACC_B) -> in
    (dtype:fp16/int32) -> iOperands (dtype:fp16/int32) -> COPY_OUT
    这种情况是COPY_OUT需要根据in的producer的iOperands来进行判断，所以会需要获取到in的producer的iOperands的数据类型。
    */
    if (TryPadMatmulMXScene(op, in)) {
        return;
    }
    if (in->tensor->rawshape.size() < MATMUL_MIN_SHAPE_SIZE) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Matmul Op %d %s input %d raw shape size is less than 2; Please check the input size.",
                          op.opmagic, op.GetOpcodeStr().c_str(), in->magic);
        return;
    }
    SetOriRawshape(in); // 先设置，再使用
    if (isL1ConvertScene) {
        /*
        输入带bias或fixpipe场景，切分tileShape为[1, N]，在L1_TO_BT和L1_TO_FIX_QUANT_PRE时，BT统一为FP32，BT
        BUFFER要求64B对齐，FixPipe为uint64，FB BUFFER为128B对齐，均要求N满足16元素对齐，否则会出现address misalign异常
        示例场景：biasShape = [1, 3208]，tileShapeN = [32, 128]，3208 % 32 = 8尾块非对齐
        另外，bias或fixpipe场景只做低维16元素对齐，高维保持不变
        Before:
        L1_TO_L0A --> L0A (shape:[24, 400]) ------------------------->    \
        L1_TO_BT --> bias_BT (shape:[1, 8]) -----> (address misalign)  A_MUL_B
        L1_TO_L0B --> L0B (shape:[400, 16]) ------------------------->    /

        After:
        L1_TO_L0A --> L0A (shape:[32, 400])  -->   \
        L1_TO_BT --> bias_BT (shape:[1, 16]) --> A_MUL_B --> output(shape:[32, 16])
        L1_TO_L0B --> L0B (shape:[400, 16])  -->   /
        */
        PadMatmulL1ConvertScene(op, in, lowIndex);
    } else {
        // GEMV 的 UB_COPY_L1(isGemv) 输入按 K 对齐、M 不 pad，避免改 RESHAPE 输出 rawshape 触发误插 spill。
        if (IsGemvL1ATensor(in) || IsGemvL0ATensor(in) || isGemvUbCopyL1) {
            PadMatmulGemvKAlign(in, highIndex, lowIndex);
        } else {
            PadMatmulHighLow(in, highIndex, lowIndex, GetMatmulPaddingValue(op, in));
        }
        TryPadMatmulIsMXScene(op, in);
    }
    if (isUB2L1Scene) {
        // GEMV 跳过 ND2NZ，无 bank 冲突优化需求，跳过 +=1（避免改 RESHAPE 输出 rawshape）。
        if (!isGemvUbCopyL1) {
            (in->tensor->rawshape[highIndex]) += 1;
        }
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "####### %d %d set rawshape as %s\n", in->tensor->rawmagic, in->magic,
                      IntVecToStr(in->tensor->rawshape).c_str());
}

bool PadLocalBuffer::IsUb2L1CopyOp(const Operation& op)
{
    if (op.iOperand.empty() || op.oOperand.empty()) {
        return false;
    }
    auto inputMemType = op.iOperand[0]->GetMemoryTypeOriginal();
    auto outputMemType = op.oOperand[0]->GetMemoryTypeOriginal();
    return (inputMemType == MemoryType::MEM_UB && outputMemType == MemoryType::MEM_L1);
}

bool PadLocalBuffer::HandleUb2L1CopyOp(Operation& op, LogicalTensorPtr& in)
{
    if (!IsUb2L1CopyOp(op)) {
        return false;
    }
    if (op.GetOpcode() != Opcode::OP_UB_COPY_L1) {
        APASS_LOG_ERROR_F(Elements::Operation, "UB to L1 copy operation expected OP_UB_COPY_L1, but got %s. %s",
                          op.GetOpcodeStr().c_str(), GetFormatBacktrace(op).c_str());
    }
    PadMatmul(op, in);
    return true;
}

bool PadLocalBuffer::ShouldSkipVectorPad(Operation& op, LogicalTensorPtr& in)
{
    if (HandleUb2L1CopyOp(op, in)) {
        return true;
    }
    if (in->shape.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Vector Op %d %s input %d shape is empty; Please check the input size. %s", op.opmagic,
                          op.GetOpcodeStr().c_str(), in->magic, GetFormatBacktrace(op).c_str());
        return true;
    }
    return false;
}

// 针对OP_CMP OP_CMPS OP_PRELU特殊OP做倒数第二轴的256B扩充
void PadLocalBuffer::PadVector256(Operation& op, LogicalTensorPtr& in, bool needRowPad)
{
    constexpr size_t kAlignBytes = 32;
    constexpr size_t kShapeSizeThres = 2;
    constexpr size_t kVectorBytes = 256;
    size_t lastDimBytes = AlignmentUtils::GetLastDimBytes(in);
    if (needRowPad && lastDimBytes != 0 && (lastDimBytes % kAlignBytes == 0)) {
        if (in->shape.size() < kShapeSizeThres || in->tensor->rawshape.size() < kShapeSizeThres) {
            if (in->shape.size() == 1 && in->tensor->rawshape.size() == 1) {
                auto vectorAlign = AlignmentUtils::GetLastDimAlignBase(in) *
                                   static_cast<int64_t>(kVectorBytes / kAlignBytes);
                in->tensor->rawshape[0] = std::max(in->tensor->rawshape[0], vectorAlign);
                APASS_LOG_INFO_F(Elements::Operation, "Op %d %s 1D input rawshape has been padded to 256B\n",
                                 op.opmagic, op.GetOpcodeStr().c_str());
            } else {
                APASS_LOG_ERROR_F(Elements::Tensor, "Tensor %d shape or rawshape dimension mismatch\n", in->GetMagic());
            }
            return;
        }

        auto dim32Count = lastDimBytes / 32;

        // 修改shape和rawshape，256B/96B非整除场景需要向上取整
        size_t lastIdx = in->shape.size() - 1;
        int64_t padValue = (8 + dim32Count - 1) / dim32Count;
        Shape& oriRawshape = GetOriRawshape(in); // 获取 DoPadding 中已设置的值
        in->tensor->rawshape[lastIdx - 1] = AlignmentUtils::PadRowDim(oriRawshape[lastIdx - 1], padValue);
        APASS_LOG_INFO_F(Elements::Operation, "Op %d %s input shape and rawshape has been changed\n", op.opmagic,
                         op.GetOpcodeStr().c_str());
    }
}

/* 1. 对于非BroadcastOp，默认做到Block对齐；
   2. 如果已经对齐到Block粒度，不做对齐---这里存在一个问题就是f16和fp32混用场景，可能对齐到一个block是不够的
   3. 对于broadcast op，如果shape小于一个Block的大小对齐到Block，否则做到两个输入之间的较大者的Block对齐。 */
void PadLocalBuffer::PadVector(Operation& op, LogicalTensorPtr& in,
                               std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw)
{
    if (ShouldSkipVectorPad(op, in)) {
        return;
    }
    OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
    int64_t paddingValue = AlignmentUtils::GetLastDimAlignBase(in); // 根据数据类型，判断需要pad到几个元素
    size_t lastIdx = in->shape.size() - 1;
    // 先设置 oriRawshape 为当前 rawshape 值，再使用
    Shape& oriRawshape = SetOriRawshape(in);
    int64_t lastDim = static_cast<int64_t>(oriRawshape[lastIdx]);
    if (calcType == OpCalcType::BROADCAST && broadcastLastAxis_.find(op.opmagic) != broadcastLastAxis_.end()) {
        lastDim = broadcastLastAxis_[op.opmagic];
    }
    int64_t shapeAfterPad = AlignmentUtils::Pad(lastDim, paddingValue);
    if (visitedRaw.count(in->tensor) == 0) {
        // shape已经对齐过，直接将rawShape对齐到shape；如果broadcast的输入是来自于view，那么整个链路上的非对齐shape都要按照
        // BROADCAST_LAST_AXIS来对齐，当前这样处理是有问题的
        in->tensor->rawshape[lastIdx] = AlignmentUtils::Pad(oriRawshape[lastIdx], shapeAfterPad);
        visitedRaw.emplace(in->tensor);
    }
}

void PadLocalBuffer::ProcessBroadcast(Operation& op, int64_t blockPadding)
{
    int64_t maxLastAxis = 0;
    bool existLessBlock = false;
    for (const auto& in : op.iOperand) {
        if (in->shape.back() <= static_cast<int>(blockPadding)) {
            existLessBlock = true;
        }
        maxLastAxis = std::max(maxLastAxis, in->shape.back());
    }
    if (!existLessBlock) {
        broadcastLastAxis_[op.opmagic] = maxLastAxis;
    }
}

void PadLocalBuffer::PrepareBroadcast(Function& function)
{
    for (auto& op : function.Operations()) {
        auto calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
        if (calcType != OpCalcType::BROADCAST) {
            continue;
        }
        int64_t blockPadding = AlignmentUtils::GetLastDimAlignBase(op.iOperand[0]);
        if (blockPadding == 1) {
            APASS_LOG_DEBUG_F(Elements::Operation, "broadcast op %d %s's datatype is not supported.", op.opmagic,
                              op.GetOpcodeStr().c_str());
            continue;
        }
        if (blockPadding <= 0) {
            continue;
        }
        ProcessBroadcast(op, blockPadding);
    }
}

bool PadLocalBuffer::IsMatmul(const LogicalTensorPtr& tensor) const
{
    auto mt = tensor->GetMemoryTypeOriginal();
    return mt == MemoryType::MEM_L1 || mt == MemoryType::MEM_L0A || mt == MemoryType::MEM_L0B ||
           mt == MemoryType::MEM_L0C || mt == MemoryType::MEM_FIX_QUANT_PRE || mt == MemoryType::MEM_BT ||
           mt == MemoryType::MEM_L0AMX || mt == MemoryType::MEM_L0BMX;
}

bool PadLocalBuffer::IsVector(const LogicalTensorPtr& tensor) const
{
    return tensor->GetMemoryTypeOriginal() == MemoryType::MEM_UB;
}

void PadLocalBuffer::PadSingleTensor(Operation& op, LogicalTensorPtr& tensor,
                                     std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw, bool needRowPad)
{
    if (IsMatmul(tensor)) {
        PadMatmul(op, tensor);
        return;
    }
    if (!IsVector(tensor) || tensor->tensor->GetRawDataSize() == 0) {
        return;
    }
    if (combineAxis_) {
        PadVectorForAxisCombine(op, tensor, visitedRaw);
    } else {
        PadVector(op, tensor, visitedRaw);
    }
    PadVector256(op, tensor, needRowPad);
}

void PadLocalBuffer::DoPadding(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visited;
    std::unordered_set<std::shared_ptr<RawTensor>> visitedRaw;
    for (auto& op : function.Operations()) {
        if (op.GetBoolAttribute("isConv"))
            continue;
        std::vector<bool> inputRowPad;
        op.GetAttr(OpAttributeKey::rowPad, inputRowPad);
        for (size_t i = 0; i < op.iOperand.size(); i++) {
            auto& in = op.iOperand[i];
            bool needRowPad = ((inputRowPad.size() > i) && inputRowPad[i]);
            if (visited.count(in) != 0) {
                if (needRowPad && IsVector(in) && in->tensor->GetRawDataSize() != 0) {
                    PadVector256(op, in, needRowPad);
                }
                continue;
            }
            visited.emplace(in);
            PadSingleTensor(op, in, visitedRaw, needRowPad);
        }
    }
    for (auto& op : function.Operations()) {
        if (op.GetBoolAttribute("isConv"))
            continue;
        for (size_t i = 0; i < op.oOperand.size(); i++) {
            auto& out = op.oOperand[i];
            if (visited.count(out) != 0)
                continue;
            visited.emplace(out);
            if (out->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            PadSingleTensor(op, out, visitedRaw);
        }
    }
}

int64_t PadLocalBuffer::AlignedRawTensorIfNeed(LogicalTensorPtr& in, int64_t pos, const int64_t base)
{
    if (in == nullptr || pos < 0 || pos >= static_cast<int64_t>(in->tensor->rawshape.size())) {
        return -1;
    }
    int64_t padDim = AlignmentUtils::Pad(in->tensor->rawshape[pos], base); // 原始代码使用 rawshape，不是 oriRawshape
    int64_t paddingValue = AlignmentUtils::GetLastDimAlignBase(in);
    if (paddingValue != 0 && padDim % paddingValue != 0) {
        padDim = std::lcm(padDim, paddingValue);
    }
    in->tensor->rawshape[pos] = padDim;
    return padDim;
}

void PadLocalBuffer::ProcessReduceForAxisCombine(Operation& op, LogicalTensorPtr& in, int64_t paddingValue)
{
    int64_t shapeSize = static_cast<int64_t>(in->shape.size());
    int64_t lastIdx = shapeSize - 1;
    int64_t padDim = AlignedRawTensorIfNeed(in, lastIdx - 1, paddingValue);
    if (op.GetOpcode() == Opcode::OP_ROWSUMLINE) {
        const auto& tempBuffer = op.GetOOperands()[1];
        size_t tempBufferLastIdx = tempBuffer->shape.size() - 1;
        tempBuffer->shape[tempBufferLastIdx] = padDim;
        tempBuffer->GetRawTensor()->rawshape[tempBufferLastIdx] = padDim;
    }
}

bool PadLocalBuffer::IsElementwiseLikeOp(OpCalcType calcType, Operation* producerOp) const
{
    if (ELEMENTWISE_LIKE_TYPES.find(calcType) != ELEMENTWISE_LIKE_TYPES.end()) {
        return true;
    }
    if (producerOp != nullptr &&
        OpcodeManager::Inst().GetOpCalcType(producerOp->GetOpcode()) == OpCalcType::BROADCAST) {
        return true;
    }
    return false;
}

void PadLocalBuffer::DoBrcbOpPadding(Operation& op, LogicalTensorPtr& in, size_t lastIdx, int64_t paddingValue,
                                     std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw)
{
    AlignedRawTensorIfNeed(in, lastIdx - 1, BRCB_SECOND_LAST_BASE);
    for (auto& out : op.GetOOperands()) {
        AlignedRawTensorIfNeed(out, lastIdx - 1, BRCB_SECOND_LAST_BASE);
        AlignedRawTensorIfNeed(out, lastIdx, paddingValue);
        visitedRaw.emplace(out->tensor);
    }
}

void PadLocalBuffer::DoElementwiseLikePadding(const Operation& op, LogicalTensorPtr& in, size_t lastIdx,
                                              int64_t paddingValue)
{
    if (op.GetOpcode() == Opcode::OP_INDEX_OUTCAST && op.GetIOperandIndex(in) == 0) {
        AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
        return;
    }
    AlignedRawTensorIfNeed(in, lastIdx - 1, paddingValue);
}

void PadLocalBuffer::PadVectorForAxisCombine(Operation& op, LogicalTensorPtr& in,
                                             std::unordered_set<std::shared_ptr<RawTensor>>& visitedRaw)
{
    if (ShouldSkipVectorPad(op, in)) {
        return;
    }
    if (visitedRaw.count(in->tensor))
        return;
    visitedRaw.emplace(in->tensor);
    OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
    int64_t paddingValue = AlignmentUtils::GetLastDimAlignBase(in);
    size_t lastIdx = in->shape.size() - 1;
    // 设置 oriRawshape（原始代码：in->tensor->oriRawshape = in->tensor->rawshape;）
    SetOriRawshape(in);
    auto producerOp = *(in->GetProducers().begin());
    bool enableAxisCombine = axisCombineMarker_.IsTensorEnableAxisCombine(in);
    bool padSecondLast = enableAxisCombine && lastIdx > 0 && in->tensor->rawshape[lastIdx] == 1;
    bool producerIsBrcb = producerOp != nullptr && producerOp->GetOpcode() == Opcode::OP_BRCB;
    if (producerIsBrcb) {
        AlignedRawTensorIfNeed(in, lastIdx - 1, BRCB_SECOND_LAST_BASE);
    }
    if (!padSecondLast) {
        AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
        if (lastIdx == 0 && producerIsBrcb) { // 防御：1D+producer为BRCB时，满足尾轴8对齐
            AlignedRawTensorIfNeed(in, lastIdx, BRCB_SECOND_LAST_BASE);
        }
        return;
    }
    if (calcType == OpCalcType::REDUCE) {
        ProcessReduceForAxisCombine(op, in, paddingValue);
        return;
    }
    if (op.GetOpcode() == Opcode::OP_BRCB) {
        DoBrcbOpPadding(op, in, lastIdx, paddingValue, visitedRaw);
        return;
    }
    if (calcType == OpCalcType::BROADCAST) {
        AlignedRawTensorIfNeed(in, lastIdx - 1, paddingValue);
        return;
    }
    if (shapeTransformOps.count(op.GetOpcode()) != 0) {
        AlignedRawTensorIfNeed(in, lastIdx - 1, paddingValue);
        return;
    }
    if (IsElementwiseLikeOp(calcType, producerOp)) {
        DoElementwiseLikePadding(op, in, lastIdx, paddingValue);
        return;
    }
    AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
}

Status PadLocalBuffer::RunOnFunction(Function& function)
{
    combineAxis_ = function.paramConfigs_.combineAxis;
    APASS_LOG_INFO_F(Elements::Operation, "======> Start PadLocalBuffer in COMBINE_AXIS=%d mode.", combineAxis_);
    oriRawshapeMap_.clear(); // 清空原始 rawshape 存储映射
    broadcastLastAxis_.clear();
    if (combineAxis_) {
        axisCombineMarker_.Run(function);
    } else {
        PrepareBroadcast(function);
    }
    DoPadding(function);
    APASS_LOG_INFO_F(Elements::Operation, "======> End PadLocalBuffer.");
    return SUCCESS;
}
} // namespace npu::tile_fwk
