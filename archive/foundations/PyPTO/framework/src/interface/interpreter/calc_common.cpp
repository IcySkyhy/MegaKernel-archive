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
 * \file calc_common.cpp
 * \brief
 */

#include "utils/string_utils.h"
#include "interface/interpreter/function.h"
#include "interface/utils/common.h"
#include "interface/interpreter/interpreter_log.h"
#include "tilefwk/data_type.h"
#include "interface/interpreter/calculator/dtype_utils.h"
#include "interface/interpreter/operation.h"
#include "interface/operation/operation_impl.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
void ExecuteOpAssemble(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() <= NUM_VALUE_2);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    // InplaceProcess 优化后，Assemble 的输入输出可能已共享同一底层 RawTensor（RawMagic
    // 相同），该op也不会被翻译成任何硬件指令； 此时无需再执行 View/Copy，直接跳过即可。
    if (ctx->op->GetIOperands()[0]->tensor->GetRawMagic() == ctx->op->GetOOperands()[0]->tensor->GetRawMagic()) {
        return;
    }
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);

    auto assemble = std::static_pointer_cast<AssembleOpAttribute>(ctx->op->GetOpAttribute());
    std::vector<int64_t> offset = ctx->opInter->EvaluateOffset(assemble->GetToOffset(), assemble->GetToDynOffset());
    auto ret = oop->View(iop->GetShape(), offset);
    if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "atomic_add")) {
        calc::Add(ret, iop, ret);
    } else {
        calc::Copy(ret, iop);
    }
    if (iop->IsShmTensor()) {
        oop->GetData()->SetAsShmTensor();
        oop->GetData()->SetShmOffset(iop->GetData()->GetShmOffset() +
                                     ret->GetStorageOffset() * ret->GetData()->GetElementSize());
    }
}
REGISTER_CALC_OP(OP_ASSEMBLE, Opcode::OP_ASSEMBLE, ExecuteOpAssemble);
REGISTER_CALC_OP(OP_CONTRACT, Opcode::OP_CONTRACT, ExecuteOpAssemble);
REGISTER_CALC_OP(OP_ASSEMBLE_SSA, Opcode::OP_ASSEMBLE_SSA, ExecuteOpAssemble);

void ExecuteOpAtomicRMW(ExecuteOperationContext* ctx)
{
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);

    auto assemble = std::static_pointer_cast<AssembleOpAttribute>(ctx->op->GetOpAttribute());
    std::vector<int64_t> offset = ctx->opInter->EvaluateOffset(assemble->GetToOffset(), assemble->GetToDynOffset());
    auto ret = oop->View(iop->GetShape(), offset);
    auto mode = (AtomicRMWMode)ctx->op->GetIntAttribute(OpAttributeKey::rmwMode);
    if (mode == AtomicRMWMode::ADD) {
        calc::Add(ret, iop, ret);
    } else if (mode == AtomicRMWMode::MAX) {
        calc::Max(ret, iop, ret);
    } else if (mode == AtomicRMWMode::MIN) {
        calc::Min(ret, iop, ret);
    } else {
        ASSERT(false) << "AtomicRMWMode not supported";
    }
}
REGISTER_CALC_OP(OP_ATOMIC_RMW, Opcode::OP_ATOMIC_RMW, ExecuteOpAtomicRMW);

void ExecuteOpNone(ExecuteOperationContext* ctx) { (void)ctx; }

void ExecuteOpInterpreterCvSyncSet(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx->opInter != nullptr);
    ctx->opInter->InterpreterSyncSimSet(ctx->op->GetSyncQueue(), ctx->op->GetOpMagic());
}

void ExecuteOpInterpreterCvSyncWait(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx->opInter != nullptr);
    ctx->opInter->InterpreterSyncSimWait(ctx->op->GetSyncQueue(), ctx->op->GetOpMagic());
}

REGISTER_CALC_OP(OP_PHASE1, Opcode::OP_PHASE1, ExecuteOpNone);
REGISTER_CALC_OP(OP_PHASE2, Opcode::OP_PHASE2, ExecuteOpNone);
REGISTER_CALC_OP(OP_SYNC_SRC, Opcode::OP_SYNC_SRC, ExecuteOpNone);
REGISTER_CALC_OP(OP_SYNC_DST, Opcode::OP_SYNC_DST, ExecuteOpNone);
REGISTER_CALC_OP(OP_BAR_V, Opcode::OP_BAR_V, ExecuteOpNone);
REGISTER_CALC_OP(OP_BAR_M, Opcode::OP_BAR_M, ExecuteOpNone);
REGISTER_CALC_OP(OP_BAR_ALL, Opcode::OP_BAR_ALL, ExecuteOpNone);
REGISTER_CALC_OP(OP_NOP, Opcode::OP_NOP, ExecuteOpNone);
REGISTER_CALC_OP(OP_CV_SYNC_SRC, Opcode::OP_CV_SYNC_SRC, ExecuteOpInterpreterCvSyncSet);
REGISTER_CALC_OP(OP_CV_SYNC_DST, Opcode::OP_CV_SYNC_DST, ExecuteOpInterpreterCvSyncWait);

void ExecuteOpViewType(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);

    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_VIEW_NULL, oop != nullptr);
    ASSERT(ExecuteOperationScene::CTX_INPUT_VIEW_NULL, iop != nullptr);

    auto opAttr = std::static_pointer_cast<ViewOpAttribute>(ctx->op->GetOpAttribute());
    // Special path: when view-type targets MEM_BT, treat it as element-wise cast.
    if (opAttr != nullptr && opAttr->GetTo() == MemoryType::MEM_BT) {
        calc::Cast(oop, iop);
        return;
    }

    auto inData = iop->GetData();
    auto outData = oop->GetData();
    if (inData->GetRawData() == outData->GetRawData()) {
        return;
    }

    // OP_VIEW_TYPE 专属：要求输入输出底层 RawTensorData 都存在
    ASSERT(ExecuteOperationScene::VIEWTYPE_BYTES_MISMATCH, inData != nullptr && outData != nullptr);

    const int64_t inElemSize = inData->GetElementSize();
    const int64_t outElemSize = outData->GetElementSize();
    const int64_t inRegionElems = iop->GetSize();
    const int64_t outRegionElems = oop->GetSize();

    // VIEW_TYPE 语义：保持底层字节数一致，只改变逻辑数据类型和 shape。
    const int64_t srcBytes = inRegionElems * inElemSize;
    const int64_t dstBytes = outRegionElems * outElemSize;
    ASSERT(ExecuteOperationScene::VIEWTYPE_BYTES_MISMATCH, srcBytes == dstBytes);

    const int64_t srcOffsetBytes = static_cast<int64_t>(iop->GetStorageOffset()) * inElemSize;
    const int64_t dstOffsetBytes = static_cast<int64_t>(oop->GetStorageOffset()) * outElemSize;

    uint8_t* srcPtr = inData->data() + srcOffsetBytes;
    uint8_t* dstPtr = outData->data() + dstOffsetBytes;

    StringUtils::DataCopy(dstPtr, dstBytes, srcPtr, srcBytes);
}
REGISTER_CALC_OP(OP_VIEW_TYPE, Opcode::OP_VIEW_TYPE, ExecuteOpViewType);

static void CheckViewToDynValidShapeNotExceedInput(const std::vector<int64_t>& toDynValidShape,
                                                   const std::vector<int64_t>& inputValidShape)
{
    if (toDynValidShape.empty()) {
        return;
    }
    const size_t dimCount = std::min(toDynValidShape.size(), inputValidShape.size());
    for (size_t i = 0; i < dimCount; ++i) {
        ASSERT(ExecuteOperationScene::VIEW_VALIDSHAPE_EXCEEDS_INPUT, toDynValidShape[i] <= inputValidShape[i])
            << "View is illegal: toDynValidShape " << IntVecToStr(toDynValidShape) << " exceeds input validShape "
            << IntVecToStr(inputValidShape);
    }
}

void ExecuteOpView(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    ASSERT(ExecuteOperationScene::CTX_NULL, oop != nullptr);
    ASSERT(ExecuteOperationScene::CTX_INPUT_VIEW_NULL, iop != nullptr);

    // 若输入输出 dtype 不同，则按 ViewType 语义处理（保持底层字节不变，仅视图变换）。
    // ViewType 会按字节比缩放最后一维，toDynValidShape 与输入 validShape 不可直接比较，且该路径可能没有
    // ViewOpAttribute。
    auto iopDataType = ctx->op->GetIOperands()[0]->GetRawTensor()->GetDataType();
    auto oopDataType = ctx->op->GetOOperands()[0]->GetRawTensor()->GetDataType();
    if (iopDataType != oopDataType) {
        ExecuteOpViewType(ctx);
        return;
    }

    auto opAttr = std::static_pointer_cast<ViewOpAttribute>(ctx->op->GetOpAttribute());
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, opAttr != nullptr);
    auto offset = ctx->opInter->EvaluateOffset(opAttr->GetFromOffset(), opAttr->GetFromDynOffset());
    std::vector<int64_t> validShape;
    if (ctx->frame->callopAttr != nullptr) {
        auto& linearArgList = ctx->frame->callopAttr->GetLinearArgList();
        validShape = ctx->opInter->EvaluateValidShape(opAttr->GetToDynValidShape(), linearArgList);
    } else {
        validShape = ctx->opInter->EvaluateValidShape(opAttr->GetToDynValidShape());
    }
    if (ctx->frame->verifyType == VerifyType::TENSOR_GRAPH) {
        CheckViewToDynValidShapeNotExceedInput(validShape, iop->GetValidShape());
    }

    if (oop->GetData() == iop->GetData()) {
        return;
    }
    uint64_t shmOffset = 0;

    bool trans = (ctx->op->HasAttr(Matrix::L1_TO_L0_TRANSPOSE)) ?
                     ctx->op->GetBoolAttribute(Matrix::L1_TO_L0_TRANSPOSE) :
                     false;
    bool isMx = false;
    if (ctx->op->HasAttr(Matrix::A_MUL_B_COPY_IN_MODE)) {
        trans = (ctx->op->GetIntAttribute(Matrix::A_MUL_B_COPY_IN_MODE) ==
                 static_cast<int64_t>(Matrix::CopyInMode::DN2NZ)) ?
                    true :
                    false;
        isMx = true;
    }
    if (trans) {
        std::vector<int64_t> oop_trans = oop->GetShape();
        ASSERT(ExecuteOperationScene::INVALID_TENSOR_SIZE, oop->GetShape().size() >= SHAPE_DIM2);
        std::swap(oop_trans[0], oop_trans[1]);
        auto ret = iop->View(oop_trans, offset);
        calc::Copy(oop, ret, trans, isMx);
        shmOffset = ret->GetStorageOffset() * ret->GetData()->GetElementSize();
    } else {
        auto ret = iop->View(validShape, offset);
        calc::Copy(oop, ret);
        shmOffset = ret->GetStorageOffset() * ret->GetData()->GetElementSize();
    }
    if (iop->IsShmTensor()) {
        oop->GetData()->SetAsShmTensor();
        oop->GetData()->SetShmOffset(iop->GetData()->GetShmOffset() + shmOffset);
    }
}
REGISTER_CALC_OP(OP_VIEW, Opcode::OP_VIEW, ExecuteOpView);
REGISTER_CALC_OP(OP_SLICE, Opcode::OP_SLICE, ExecuteOpView);

void ExecuteOpCopyOut(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() <= NUM_VALUE_2);
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    ASSERT(ExecuteOperationScene::CTX_INPUT_VIEW_NULL, iop != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_VIEW_NULL, oop != nullptr);

    auto copyout = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
    auto [from, toOffsetAttr] = copyout->GetCopyOutAttr();
    std::vector<int64_t> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyout->GetShape());
    std::vector<int64_t> rawShape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyout->GetRawShape());
    std::vector<int64_t> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, toOffsetAttr);

    std::vector<int64_t> iopShape = iop->GetShape();
    if (oop->GetIsSpilled()) {
        std::fill(toOffset.begin(), toOffset.end(), 0);
    }

    bool axisCombine = ctx->op->GetBoolAttribute("input_combine_axis");
    auto oopValid = std::make_shared<LogicalTensorData>(oop->GetData(), iopShape, toOffset);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_VIEW_NULL, oopValid != nullptr);
    if (from == MemoryType::MEM_L0C) {
        if (iop->GetDataType() == DataType::DT_INT32 && oop->GetDataType() == DataType::DT_FP16) {
            uint64_t scale = (ctx->op->HasAttr(Matrix::A_MUL_B_SCALE_ATTR)) ?
                                 ctx->op->GetElementAttribute(Matrix::A_MUL_B_SCALE_ATTR).GetUnsignedData() :
                                 0;
            int relu = (ctx->op->HasAttr(Matrix::A_MUL_B_RELU_ATTR)) ?
                           ctx->op->GetIntAttribute(Matrix::A_MUL_B_RELU_ATTR) :
                           0;
            LogicalTensorDataPtr scalePtr = nullptr;
            if (ctx->ioperandDataViewList->size() > 1) {
                scalePtr = ctx->ioperandDataViewList->at(1);
            }
            calc::QuantPreCompute(oopValid, iop, scalePtr, scale, relu);
            return;
        }
        if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "atomic_add")) {
            calc::Add(oopValid, iop, oopValid);
        } else {
            calc::Cast(oopValid, iop);
        }
    } else if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "atomic_add")) {
        calc::Add(oopValid, iop, oopValid);
    } else {
        calc::Copy(oopValid, iop, axisCombine);
    }
}
REGISTER_CALC_OP(OP_COPY_OUT, Opcode::OP_COPY_OUT, ExecuteOpCopyOut);

void ExecuteOpCopyIn(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);

    auto copyin = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());

    bool outputCombineAxisDone = ctx->op->GetBoolAttribute("input_combine_axis_done");
    std::vector<int64_t> oopShape = oop->GetShape();
    std::vector<int> axises = {0, 1};
    LogicalTensorDataPtr oopTrans;
    if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
        std::vector<int64_t> transShape = {oopShape[1], oopShape[0]};
        std::vector<int64_t> transRawShape = {oop->GetData()->GetShape()[1], oop->GetData()->GetShape()[0]};
        oopTrans = LogicalTensorData::CreateEmpty(oop->GetDataType(), transShape, std::vector<int64_t>(0),
                                                  transRawShape);
    }

    // HACK: copyin's default attribute should be full tensor
    auto iopValid = iop;
    auto oopValid = oop;
    if (copyin != nullptr) {
        std::vector<int64_t> rawShape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetRawShape());
        std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
        std::vector<int64_t> dynvalidshape = ctx->opInter->EvaluateOpImmediate(ctx->frame,
                                                                               copyin->GetToDynValidShape());
        if (dynvalidshape.empty()) {
            dynvalidshape = iopValid->GetValidShape();
        }

        iopValid = std::make_shared<LogicalTensorData>(iopValid->GetData(), dynvalidshape, fromOffset);
        if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
            oopTrans = oopTrans->View(dynvalidshape, std::vector<int64_t>(fromOffset.size(), 0));
        }
    }

    if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
        calc::Copy(oopValid, iopValid, true);
    } else {
        calc::Copy(oopValid, iopValid);
    }
}
REGISTER_CALC_OP(OP_COPY_IN, Opcode::OP_COPY_IN, ExecuteOpCopyIn);

void ExecuteOpCopy(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    calc::Copy(oop, iop);
}
REGISTER_CALC_OP(OP_REGISTER_COPY, Opcode::OP_REGISTER_COPY, ExecuteOpCopy);

static bool NeedDecodeToFp32ForDump(DataType dtype)
{
    return dtype == DT_INT4 || IsFp8Dtype(dtype) || dtype == DT_HF4 || dtype == DT_HF8 || IsFp4PackedDtype(dtype);
}

static std::string TensorToDumpString(const LogicalTensorDataPtr& tensor)
{
    if (tensor == nullptr) {
        return "???";
    }
    if (!NeedDecodeToFp32ForDump(tensor->GetDataType())) {
        return tensor->ToString();
    }
    auto shape = tensor->GetShape();
    auto castOut = LogicalTensorData::CreateEmpty(DT_FP32, shape, shape, shape);
    calc::Cast(castOut, tensor);
    return castOut->ToString();
}

std::string FormatString(const std::string& s, OperationInterpreter* opInter,
                         const std::vector<LogicalTensorDataPtr>* iopDataView,
                         const std::vector<SymbolicScalar>* scalars)
{
    std::stringstream ss;
    size_t pos = 0;
    size_t tpos = 0, spos = 0;
    while (pos < s.size()) {
        if (s[pos] == '{') {
            size_t end = s.find('}', pos + 1);
            if (end == std::string::npos) {
                ss << s.substr(pos);
                break;
            }
            auto symbol = s.substr(pos + 1, end - pos - 1);
            if (symbol == "T") {
                if (iopDataView && tpos < iopDataView->size())
                    ss << TensorToDumpString((*iopDataView)[tpos++]);
                else
                    ss << "???";
            } else if (symbol == "S") {
                if (scalars && spos < scalars->size())
                    ss << opInter->EvaluateSymbolicScalar((*scalars)[spos++]);
                else
                    ss << "???";
            } else {
                ss << '{' << symbol << '}';
            }
            pos = end + 1;
        } else {
            ss << s[pos];
            pos++;
        }
    }
    return ss.str();
}

void ExecutePrint(ExecuteOperationContext* ctx)
{
    auto cond = ctx->op->GetSymbolicScalarAttribute(OP_ATTR_PREFIX + "cond");
    if (!ctx->opInter->EvaluateSymbolicScalar(cond)) {
        return;
    }

    auto scalars = ctx->op->GetAttr<std::vector<SymbolicScalar>>(OP_ATTR_PREFIX + "scalars");
    if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "fname")) {
        auto fname = ctx->op->GetStringAttribute(OP_ATTR_PREFIX + "fname");
        uint64_t ts = 0;
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        ts = (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
        auto baseName = FormatString(fname, ctx->opInter, nullptr, scalars);
        auto basePath = config::LogTopFolder() + "/tensor/" + baseName + "_" + std::to_string(ts);
        auto binPath = basePath + ".data";
        auto csvPath = basePath + ".csv";
        auto& iop = ctx->ioperandDataViewList->at(0);
        auto shape = iop->GetValidShape();
        if (shape.empty()) {
            shape = iop->GetShape();
        }
        auto oop = LogicalTensorData::CreateEmpty(iop->GetDataType(), shape, shape, iop->GetData()->GetShape());
        calc::Copy(oop, iop);
        oop->GetData()->ToFile(binPath);
        std::ofstream csv(csvPath, std::ios::out);
        if (csv) {
            csv << "key,value\n";
            csv << "dtype," << static_cast<int>(oop->GetDataType()) << "\n";
            csv << "shape,";
            if (!shape.empty()) {
                csv << shape[0];
                for (size_t i = 1; i < shape.size(); i++) {
                    csv << "x" << shape[i];
                }
            }
            csv << "\n";
            csv << "element_count," << oop->GetData()->GetSize() << "\n";
            csv.close();
        } else {
            INTERPRETER_LOGE_FULL(OpDumpScene::DUMP_OPEN_FILE_FAILED, "open csv file %s failed!!!!", csvPath.c_str());
        }
    }

    std::string format;
    if (ctx->op->GetAttr(OP_ATTR_PREFIX + "format", format)) {
        std::cout << FormatString(format, ctx->opInter, ctx->ioperandDataViewList, scalars) << std::endl;
    }
}
REGISTER_CALC_OP(OP_PRINT, Opcode::OP_PRINT, ExecutePrint);

void ExecuteOpReshape(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto& oop = ctx->ooperandInplaceDataViewList->at(0);
    auto& iop = ctx->ioperandDataViewList->at(0);
    int64_t iopSize = std::accumulate(iop->GetValidShape().begin(), iop->GetValidShape().end(), (int64_t)1,
                                      std::multiplies<int64_t>());
    int64_t oopSize = std::accumulate(oop->GetValidShape().begin(), oop->GetValidShape().end(), (int64_t)1,
                                      std::multiplies<int64_t>());
    if (ctx->op->GetOOperands()[0]->GetRawMagic() == ctx->op->GetIOperands()[0]->GetRawMagic() && iopSize != oopSize) {
        return;
    }
    auto iopDataView = iop->View(iop->GetValidShape(), iop->GetOffset());
    auto oopDataView = oop->View(oop->GetValidShape(), oop->GetOffset());
    if (oop->GetData()->GetSize() > iop->GetData()->GetSize()) {
        auto oopRawShapeStr = IntVecToStr(oop->GetData()->GetShape());
        auto iopRawShapeStr = IntVecToStr(iop->GetData()->GetShape());
        INTERPRETER_LOGW("Reshape: rawTensor shape/size mismatch in padding path: "
                         "oop(raw) shape=%s size=%ld, iop(raw) shape=%s size=%ld",
                         oopRawShapeStr.c_str(), oop->GetData()->GetSize(), iopRawShapeStr.c_str(),
                         iop->GetData()->GetSize());
        auto paddingRaw = std::make_shared<RawTensorData>(oop->GetData()->GetDataType(), iop->GetShape());
        auto paddingIopView = std::make_shared<LogicalTensorData>(paddingRaw, iop->GetValidShape(),
                                                                  iop->GetValidShape(), iop->GetOffset());
        calc::Copy(paddingIopView, iopDataView);
        iopDataView = paddingIopView;
    }
    auto actualIop = std::make_shared<LogicalTensorData>(iopDataView->GetData());
    auto actualOop = std::make_shared<LogicalTensorData>(oopDataView->GetData());
    if (oopDataView->GetSize() > iopDataView->GetSize()) {
        INTERPRETER_LOGW("%s", ctx->op->Dump().c_str());
        INTERPRETER_LOGW("iop validShape: %s ---> oop validShape: %s", IntVecToStr(iop->GetShape()).c_str(),
                         IntVecToStr(oop->GetShape()).c_str());
        INTERPRETER_LOGW("Reshape: input tensor is not enough to reshape to output tensor");
        calc::Reshape(oopDataView, actualIop);
    } else if (oopDataView->GetSize() < iopDataView->GetSize()) {
        calc::Reshape(actualOop, iopDataView);
    } else {
        calc::Reshape(oopDataView, iopDataView);
    }
}
REGISTER_CALC_OP(OP_RESHAPE, Opcode::OP_RESHAPE, ExecuteOpReshape);
REGISTER_CALC_OP(OP_RESHAPE_COPY_OUT, Opcode::OP_RESHAPE_COPY_OUT, ExecuteOpCopyOut);
REGISTER_CALC_OP(OP_L0C_RESHAPE_COPY_OUT, Opcode::OP_L0C_RESHAPE_COPY_OUT, ExecuteOpCopyOut);
REGISTER_CALC_OP(OP_RESHAPE_COPY_IN, Opcode::OP_RESHAPE_COPY_IN, ExecuteOpCopyIn);
REGISTER_CALC_OP(OP_L1_RESHAPE_COPY_IN, Opcode::OP_L1_RESHAPE_COPY_IN, ExecuteOpCopyIn);
} // namespace npu::tile_fwk
