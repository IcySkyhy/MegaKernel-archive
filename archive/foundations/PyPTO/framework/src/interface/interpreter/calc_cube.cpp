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
 * \file calc_cube.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/interpreter/operation.h"
#include "interface/operation/operation_impl.h"
#include "tilefwk/error_code.h"
#include "interface/utils/common.h"

using namespace npu::tile_fwk::calc;

namespace npu::tile_fwk {

namespace {
constexpr int64_t DN2NZ_MODE = static_cast<int64_t>(Matrix::CopyInMode::DN2NZ);
constexpr size_t SCALE_A_INDEX = 2;
constexpr size_t SCALE_B_INDEX = 3;
constexpr size_t SCALE_A_INDEX_ACC = 3;
constexpr size_t SCALE_B_INDEX_ACC = 4;
constexpr size_t BIAS_DEFAULT_INDEX = 2;
constexpr size_t BIAS_WITH_SCALE_INDEX = 4;

bool IsAccOp(Opcode opcode) { return opcode == Opcode::OP_A_MULACC_B || opcode == Opcode::OP_A_MULACC_BT; }

bool IsMxScaleTensor(const LogicalTensorDataPtr& tensor)
{
    if (tensor == nullptr || tensor->GetDataType() != DataType::DT_FP8E8M0) {
        return false;
    }
    const auto& shape = tensor->GetShape();
    return shape.size() == 0x3 && shape[0x3 - 1] == 0x2;
}

bool HasMxScaleByInputs(const ExecuteOperationContext* ctx, bool isAccOp)
{
    return !isAccOp && ctx->ioperandDataViewList->size() >= BIAS_WITH_SCALE_INDEX &&
           IsMxScaleTensor(ctx->ioperandDataViewList->at(SCALE_A_INDEX)) &&
           IsMxScaleTensor(ctx->ioperandDataViewList->at(SCALE_B_INDEX));
}

bool GetBoolAttrOrDefault(const Operation* op, const std::string& attr, bool defaultValue = false)
{
    return op->HasAttr(attr) ? op->GetBoolAttribute(attr) : defaultValue;
}

int GetIntAttrOrDefault(const Operation* op, const std::string& attr, int defaultValue = 0)
{
    return op->HasAttr(attr) ? op->GetIntAttribute(attr) : defaultValue;
}

uint64_t GetScaleAttrOrDefault(const Operation* op)
{
    return op->HasAttr(Matrix::A_MUL_B_SCALE_ATTR) ?
               op->GetElementAttribute(Matrix::A_MUL_B_SCALE_ATTR).GetUnsignedData() :
               0;
}

bool HasMXScaleValue(const ExecuteOperationContext* ctx)
{
    if (ctx->ioperandDataViewList->size() <= SCALE_B_INDEX) {
        return false;
    }

    for (size_t idx = 0; idx < ctx->ioperandDataViewList->size(); idx++) {
        auto tensor = ctx->ioperandDataViewList->at(idx);
        // 存在输入为FP8E8M0的tensor时，说明该tensor为mxscale
        if (tensor != nullptr && tensor->GetDataType() == DataType::DT_FP8E8M0) {
            return true;
        }
    }
    return false;
}

LogicalTensorDataPtr FindFixpipeScaleInput(const ExecuteOperationContext* ctx, bool isAccOp)
{
    const size_t startIdx = isAccOp ? 3 : 2;
    for (size_t idx = startIdx; idx < ctx->ioperandDataViewList->size(); idx++) {
        auto tensor = ctx->ioperandDataViewList->at(idx);
        if (tensor != nullptr && tensor->GetDataType() == DataType::DT_UINT64) {
            return tensor;
        }
    }
    return nullptr;
}

MatMulParam BuildMatMulParam(const ExecuteOperationContext* ctx, bool hasMXScale, LogicalTensorDataPtr bias,
                             LogicalTensorDataPtr aScale, LogicalTensorDataPtr bScale, LogicalTensorDataPtr scalePtr)
{
    bool transAScale = hasMXScale && ctx->op->HasAttr(Matrix::A_MUL_B_SCALE_A_COPY_IN_MODE) &&
                       ctx->op->GetIntAttribute(Matrix::A_MUL_B_SCALE_A_COPY_IN_MODE) == DN2NZ_MODE;
    bool transBScale = hasMXScale && ctx->op->HasAttr(Matrix::A_MUL_B_SCALE_B_COPY_IN_MODE) &&
                       ctx->op->GetIntAttribute(Matrix::A_MUL_B_SCALE_B_COPY_IN_MODE) == DN2NZ_MODE;

    auto& cubeTile = ctx->op->GetTileShape().GetCubeTile();
    int k1 = cubeTile.k[1];
    int k2 = cubeTile.k[2];
    int kStep = std::gcd(k1, k2);

    bool transA = GetBoolAttrOrDefault(ctx->op, Matrix::A_MUL_B_TRANS_A);
    bool transB = GetBoolAttrOrDefault(ctx->op, Matrix::A_MUL_B_TRANS_B);
    uint64_t scale = GetScaleAttrOrDefault(ctx->op);
    int relu = GetIntAttrOrDefault(ctx->op, Matrix::A_MUL_B_RELU_ATTR);

    MatMulParam param = {transA, transB,  transAScale, transBScale, kStep,  scale,
                         relu,   nullptr, nullptr,     nullptr,     nullptr};

    // 为每个可能的临时对象分配堆内存，由调用者负责清理
    if (scalePtr != nullptr) {
        param.scalePtr = new TensorData(Trans(scalePtr));
    }

    if (bias != nullptr) {
        param.biasPtr = new TensorData(Trans(bias));
    }

    if (aScale != nullptr) {
        param.aScalePtr = new TensorData(Trans(aScale));
    }

    if (bScale != nullptr) {
        param.bScalePtr = new TensorData(Trans(bScale));
    }

    return param;
}

void CleanupMatMulParam(MatMulParam& param)
{
    // 清理动态分配的内存
    if (param.scalePtr) {
        delete param.scalePtr;
        param.scalePtr = nullptr;
    }
    if (param.biasPtr) {
        delete param.biasPtr;
        param.biasPtr = nullptr;
    }
    if (param.aScalePtr) {
        delete param.aScalePtr;
        param.aScalePtr = nullptr;
    }
    if (param.bScalePtr) {
        delete param.bScalePtr;
        param.bScalePtr = nullptr;
    }
}
} // namespace

void ExecuteOpAMulB(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx->ioperandDataViewList != nullptr);
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx->ooperandInplaceDataViewList != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto lhs = ctx->ioperandDataViewList->at(0);
    auto rhs = ctx->ioperandDataViewList->at(1);
    Opcode opcode = ctx->op->GetOpcode();
    bool isAccOp = IsAccOp(opcode);
    bool hasMXScaleByInputs = HasMxScaleByInputs(ctx, isAccOp);
    bool hasMXScale = HasMXScaleValue(ctx);
    bool hasBias = GetBoolAttrOrDefault(ctx->op, Matrix::A_MUL_B_BIAS_ATTR) ||
                   (!isAccOp && hasMXScaleByInputs && ctx->ioperandDataViewList->size() > BIAS_WITH_SCALE_INDEX);
    size_t biasIndex = BIAS_DEFAULT_INDEX;
    if (!isAccOp && hasMXScale) {
        // scaled_mm interface: [A, B, scale_a, scale_b, (optional) bias]
        biasIndex = BIAS_WITH_SCALE_INDEX;
    }
    auto bias = (hasBias && ctx->ioperandDataViewList->size() > biasIndex) ? ctx->ioperandDataViewList->at(biasIndex) :
                                                                             nullptr;
    // scaled_mm interface without acc op: [A, B, scale_a, scale_b]
    // scaled_mm interface with acc op: [A, B, acc, scale_a, scale_b]
    size_t indexAScale = isAccOp ? SCALE_A_INDEX_ACC : SCALE_A_INDEX;
    size_t indexBScale = isAccOp ? SCALE_B_INDEX_ACC : SCALE_B_INDEX;
    auto aScale = (hasMXScale && ctx->ioperandDataViewList->size() > indexAScale) ?
                      ctx->ioperandDataViewList->at(indexAScale) :
                      nullptr;
    auto bScale = (hasMXScale && ctx->ioperandDataViewList->size() > indexBScale) ?
                      ctx->ioperandDataViewList->at(indexBScale) :
                      nullptr;
    LogicalTensorDataPtr scalePtr = nullptr;
    if (lhs->GetDataType() == DataType::DT_INT8 && ret->GetDataType() == DataType::DT_FP16) {
        scalePtr = FindFixpipeScaleInput(ctx, isAccOp);
    }
    MatMulParam param = BuildMatMulParam(ctx, hasMXScale, bias, aScale, bScale, scalePtr);
    switch (opcode) {
        case Opcode::OP_A_MUL_B: {
            calc::MatMul(ret, lhs, rhs, param);
        } break;
        case Opcode::OP_A_MULACC_B: {
            auto acc = ctx->ioperandDataViewList->at(2);
            calc::AccMatMul(ret, lhs, rhs, acc, param);
        } break;
        default:
            ASSERT(ExecuteOperationScene::UNSUPPORTED_OPCODE, false);
            break;
    }
    // 清理动态分配的内存
    CleanupMatMulParam(param);
}

REGISTER_CALC_OP(OP_A_MUL_B, Opcode::OP_A_MUL_B, ExecuteOpAMulB);
REGISTER_CALC_OP(OP_A_MULACC_B, Opcode::OP_A_MULACC_B, ExecuteOpAMulB);

void ExecuteOpAlloc(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() <= 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 0);
}

REGISTER_CALC_OP(OP_UB_ALLOC, Opcode::OP_UB_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_L1_ALLOC, Opcode::OP_L1_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_FIX_ALLOC, Opcode::OP_FIX_ALLOC, ExecuteOpAlloc);
REGISTER_CALC_OP(OP_BT_ALLOC, Opcode::OP_BT_ALLOC, ExecuteOpAlloc);

void ExecuteL1ToL0(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& oper = ctx->ioperandDataViewList->at(0);
    ASSERT(ExecuteOperationScene::CTX_NULL, ret != nullptr);
    ASSERT(ExecuteOperationScene::INVALID_TENSOR_SIZE, ret->GetShape().size() >= SHAPE_DIM2);
    Opcode opCode = ctx->op->GetOpcode();
    bool trans = opCode == Opcode::OP_L1_TO_L0_BT || opCode == Opcode::OP_L1_TO_L0_AT;
    bool isMx = false;
    if (ctx->op->HasAttr(Matrix::A_MUL_B_COPY_IN_MODE)) {
        trans = (ctx->op->GetIntAttribute(Matrix::A_MUL_B_COPY_IN_MODE) ==
                 static_cast<int64_t>(Matrix::CopyInMode::DN2NZ)) ?
                    true :
                    false;
        isMx = true;
    }
    auto copyin = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute()); // 获取attr
    if (copyin == nullptr) {
        calc::Copy(ret, oper, trans);
        return;
    }
    std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
    if (trans) {
        std::vector<int64_t> oop_trans = ret->GetShape();
        std::swap(oop_trans[0], oop_trans[1]);
        auto iop = oper->View(oop_trans, fromOffset);
        calc::Copy(ret, iop, trans, isMx);
    } else {
        auto iop = oper->View(ret->GetShape(), fromOffset);
        calc::Copy(ret, iop, trans);
    }
}

REGISTER_CALC_OP(OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, ExecuteL1ToL0);
REGISTER_CALC_OP(OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, ExecuteL1ToL0);
REGISTER_CALC_OP(OP_L1_TO_L0_AT, Opcode::OP_L1_TO_L0_AT, ExecuteL1ToL0);
REGISTER_CALC_OP(OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0_BT, ExecuteL1ToL0);
REGISTER_CALC_OP(OP_L1_COPY_IN_A_SCALE, Opcode::OP_L1_COPY_IN_A_SCALE, ExecuteL1ToL0);
REGISTER_CALC_OP(OP_L1_COPY_IN_B_SCALE, Opcode::OP_L1_COPY_IN_B_SCALE, ExecuteL1ToL0);
REGISTER_CALC_OP(OP_L1_TO_L0A_SCALE, Opcode::OP_L1_TO_L0A_SCALE, ExecuteL1ToL0);
REGISTER_CALC_OP(OP_L1_TO_L0B_SCALE, Opcode::OP_L1_TO_L0B_SCALE, ExecuteL1ToL0);

void ExecuteL0CToL1(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& oper = ctx->ioperandDataViewList->at(0);
    auto copyin = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute()); // 获取attr
    ASSERT(ExecuteOperationScene::CTX_INPUT_VIEW_NULL, oper != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_VIEW_NULL, ret != nullptr);
    ASSERT(ExecuteOperationScene::L0C_TO_L1_SHAPE_NOT_2D,
           oper->GetShape().size() == SHAPE_DIM2 && ret->GetShape().size() == SHAPE_DIM2);
    bool quantFlag = oper->GetDataType() == DataType::DT_INT32 && ret->GetDataType() == DataType::DT_FP16;
    uint64_t scale = (ctx->op->HasAttr(Matrix::A_MUL_B_SCALE_ATTR)) ?
                         ctx->op->GetElementAttribute(Matrix::A_MUL_B_SCALE_ATTR).GetUnsignedData() :
                         0;
    int relu = (ctx->op->HasAttr(Matrix::A_MUL_B_RELU_ATTR)) ? ctx->op->GetIntAttribute(Matrix::A_MUL_B_RELU_ATTR) : 0;
    LogicalTensorDataPtr scalePtr = nullptr;
    if (ctx->ioperandDataViewList->size() > 1) {
        scalePtr = ctx->ioperandDataViewList->at(1);
    }
    std::vector<int64_t> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetShape());
    std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
    std::vector<int64_t> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetToOffset());
    if (oper->GetShape()[0] > ret->GetShape()[0] || oper->GetShape()[1] > ret->GetShape()[1]) {
        auto iop = oper->View(ret->GetShape(), fromOffset);
        if (quantFlag) {
            LogicalTensorDataPtr scaleOp = nullptr;
            if (scalePtr != nullptr) {
                scaleOp = scalePtr->View({1, ret->GetShape()[1]}, {0, fromOffset[1]});
                calc::QuantPreCompute(ret, iop, scaleOp, scale, relu);
            } else {
                calc::QuantPreCompute(ret, iop, nullptr, scale, relu);
            }
        } else {
            calc::Copy(ret, iop);
        }
    } else {
        auto iop = ret->View(oper->GetShape(), toOffset);
        if (quantFlag) {
            if (scalePtr != nullptr) {
                calc::QuantPreCompute(iop, oper, scalePtr, scale, relu);
            } else {
                calc::QuantPreCompute(iop, oper, nullptr, scale, relu);
            }
        } else {
            calc::Copy(iop, oper);
        }
    }
}

REGISTER_CALC_OP(OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1, ExecuteL0CToL1);
REGISTER_CALC_OP(OP_L0C_COPY_UB, Opcode::OP_L0C_COPY_UB, ExecuteL0CToL1);
REGISTER_CALC_OP(OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1, ExecuteL0CToL1);

void ExecuteDuplicate(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    auto& ret = ctx->ooperandInplaceDataViewList->at(0);
    auto& oper = ctx->ioperandDataViewList->at(0);
    calc::Copy(ret, oper);
}

REGISTER_CALC_OP(OP_CONVERT, Opcode::OP_CONVERT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_FIX_QUANT_PRE, Opcode::OP_L1_TO_FIX_QUANT_PRE, ExecuteDuplicate);
REGISTER_CALC_OP(OP_L1_TO_BT, Opcode::OP_L1_TO_BT, ExecuteDuplicate);
REGISTER_CALC_OP(OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_ND2NZ, ExecuteDuplicate);
REGISTER_CALC_OP(OP_UB_COPY_L1_ND, Opcode::OP_UB_COPY_L1_ND, ExecuteDuplicate);

void ExecuteOpGatherInL1(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == SIZE_THREE);
    auto output = ctx->ooperandInplaceDataViewList->at(0);
    auto params = ctx->ioperandDataViewList->at(0);
    auto indices = ctx->ioperandDataViewList->at(1);
    auto pageTable = ctx->ioperandDataViewList->at(2);
    int blocksize = ctx->op->GetIntAttribute("op_attr_blocksize");
    calc::GatherInL1(output, params, indices, pageTable, blocksize);
}

REGISTER_CALC_OP(OP_GATHER_IN_L1, Opcode::OP_GATHER_IN_L1, ExecuteOpGatherInL1);
} // namespace npu::tile_fwk
