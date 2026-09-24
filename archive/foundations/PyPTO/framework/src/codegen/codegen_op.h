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
 * \file codegen_op.h
 * \brief
 */

#ifndef CODEGEN_OP_H
#define CODEGEN_OP_H

#include <map>
#include <tuple>
#include <cstdint>
#include <string>
#include <utility>
#include <unordered_set>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/symbol_mgr/codegen_symbol.h"

namespace npu::tile_fwk {
const int MAX_OPERANDS = 11;
const int NULL_OPERAND = 0;

struct CodeGenOpCtx {
    std::shared_ptr<SymbolManager> symbolManager;
    Function& topFunc;
    Function& subFunc;
    const Operation& operation;
    bool isMainBlock{false};
    bool isDynamicAligned{false};

    CodeGenOpCtx(std::shared_ptr<SymbolManager> sm, Function& tf, Function& sf, const Operation& op,
                 bool isMainBlk = false, bool isDynAligned = false)
        : symbolManager(std::move(sm)),
          topFunc(tf),
          subFunc(sf),
          operation(op),
          isMainBlock(isMainBlk),
          isDynamicAligned(isDynAligned)
    {}
};

class CodeGenOp {
public:
    explicit CodeGenOp(const CodeGenOpCtx& ctx)
        : originalOp(ctx.operation),
          functionType(ctx.topFunc.GetFunctionType()),
          isMainBlock(ctx.isMainBlock),
          isDynamicAligned(ctx.isDynamicAligned)
    {
        operandWithMagic.reserve(MAX_OPERANDS);
        for (size_t i = 0; i < MAX_OPERANDS; i++) {
            operandWithMagic[i] = INVALID_TENSOR_MAGIC;
            operandType[i] = BUF_UNKNOWN;
            operandDtype[i] = DataType::DT_BOTTOM;
            // In COA(Call Operation Attribute), 0-index is the callee's cce info. So the tensor list starts from 1.
            paramLocation[i] = 1;
        }
        sm = ctx.symbolManager;
    }
    virtual ~CodeGenOp() = default;

    virtual void Init(const Operation& ops);

    virtual std::string GenOpCode() const = 0;

protected:
    std::string GenOpAttr(bool hasExistingParam = true) const;

    const Operation& originalOp;
    const FunctionType functionType;
    std::string opCodeStr;
    Opcode opCode{Opcode::OP_UNKNOWN};
    std::string aliasOp; // alias op name

    std::vector<int> operandWithMagic;
    OperandType operandType[MAX_OPERANDS] = {};
    DataType operandDtype[MAX_OPERANDS] = {};
    Element extOperandVal;
    SymbolicScalar extSymbolicScalar;
    std::vector<Element> extScalarVec;

    std::vector<int64_t> offset[MAX_OPERANDS] = {};
    std::vector<int64_t> shape[MAX_OPERANDS] = {};
    std::vector<int64_t> rawShape[MAX_OPERANDS] = {};
    std::vector<SymbolicScalar> dynamicRawShape[MAX_OPERANDS] = {}; // raw shape

    std::vector<SymbolicScalar> dynamicOffset[MAX_OPERANDS] = {};
    std::vector<SymbolicScalar> dynamicValidShape[MAX_OPERANDS] = {}; // valid shape

    std::vector<int64_t> shapeFromAttr[MAX_OPERANDS] = {};         // 1.for spilling to GM scene 2.for conv
    std::vector<SymbolicScalar> offsetFromAttr[MAX_OPERANDS] = {}; // for spilling to GM scene
    std::vector<SymbolicScalar> dynValidShapeFromOpAttr[MAX_OPERANDS] = {};
    // if operand is an variable, record its related argument location
    int paramLocation[MAX_OPERANDS] = {};
    int GmTensorParamIdxInCallFunc{0};
    OpSyncQueue syncQueue;

    // add for ooo sched
    int addrOffset[MAX_OPERANDS] = {};
    std::vector<long> convParams;
    std::vector<int> poolParams;

    std::map<std::string, std::any> opAttrs;
    std::map<std::string, std::any> tensorAttrs[MAX_OPERANDS];

    std::shared_ptr<SymbolManager> sm{nullptr};

    std::string tileOpName;
    bool isInputForceCombineAxis{false};
    bool isSupportDynamicAligned{false}; // NEXTNEXT delete after all TileOp is changed to TileTensor Mode
    bool isDynamicFunction{false};
    bool isSupportTileTensor{false};
    int operandCnt{0};
    bool isMainBlock{false};
    bool isDynamicAligned{false};

private:
    void UpdateCodegenOpInfoByTensor(const Operation& ops, bool isInput, const std::shared_ptr<LogicalTensor>& tensor,
                                     int& operandIdx, size_t ioIdx);

    void UpdateTileOpInfo(const Operation& ops);

    void ConvertPoolAttribute(const Operation& operation);
    void ConvertAttribute(const Operation& operation);
    void UpdateShape(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx);
    void UpdateShapeAndOffset(const Operation& ops, const LogicalTensor& logicalTensor, bool isInput, int operandIdx,
                              size_t ioIdx);
    void UpdateOffsetForInput(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx);
    void UpdateOffsetForOutput(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx);
    void UpdateOffsetValueFromAttr(const std::vector<OpImmediate>& offsets, int operandIdx);
    void UpdateScalarValue(const Operation& ops);
    void UpdateOpAttribute(const Operation& ops);
    void UpdateValidShapeForShmemCopyOps(const Operation& oper, int operandIdx, std::shared_ptr<CopyOpAttribute> attr);
    bool NeedCombineAxis(const Operation& oper, bool isInput, size_t ioIdx) const;
    void CombineAxisShape(const Operation& oper, int operandIdx);
    void CombineAxisOffset(const Operation& oper, int operandIdx);
    void UpdateDynValidShapeFromAttr(const Operation& oper, const LogicalTensor& logicalTensor, int operandIdx);
    void SetDynValidShapeFromAttr(const std::vector<OpImmediate>& toValidShape, int operandIdx);
    bool IsNeedUseNormalAddrAlloc(const Operation& ops) const;
};
} // namespace npu::tile_fwk

#endif // CODEGEN_OP_H
