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
 * \file infer_param_index.h
 * \brief
 */

#ifndef INFER_PARAM_INDEX_PASS_H_
#define INFER_PARAM_INDEX_PASS_H_
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"

namespace npu {
namespace tile_fwk {
// 设置成symxxdimxx格式的特殊op
const std::set<Opcode> setSymDimOps = {Opcode::OP_VEC_DUP, Opcode::OP_EXPAND,          Opcode::OP_RESHAPE,
                                       Opcode::OP_GATHER,  Opcode::OP_GATHER_IN_UB,    Opcode::OP_GATHER_IN_L1,
                                       Opcode::OP_PERMUTE, Opcode::OP_PERMUTE_ELEMENT, Opcode::OP_UB_COPY_L1};
// 不更改其DynValidShape的op
// Assemble、l0c copy ub和view均是因为重新推导dynvalidshape会引入错误而保留normalize
const std::set<Opcode> useSelfOps = {Opcode::OP_ASSEMBLE, Opcode::OP_L0C_COPY_UB, Opcode::OP_VIEW};

class InferParamIndex : public Pass {
public:
    InferParamIndex() : Pass("InferParamIndex") {}
    ~InferParamIndex() override {}
    Status RunOnFunction(Function& function) override;

private:
    std::string DumpParamIndex(const std::map<std::string, DynParamInfo>& dynParamTable);
    bool ResetGmCopyDynValidShape(Operation& op, Function& function);
    Status ResetOutputDynValidShape(Operation& op, Function& function);
    Status ResetViewDynValidShape(const Operation& op);
    Status ResetAssembleDynValidShape(const Operation& op);
    Status ResetDynValidShape(Function& function);
    Status InsertAddr2ValidShapeSpecified(Operation& op, std::map<int, std::vector<SymbolicScalar>>& addr2ValidShape,
                                          std::map<int, std::vector<SymbolicScalar>>& addr2ValidShapeSpecified,
                                          std::map<int, int>& magic2Coa);
    Status UpdateValidShape(Function& subFunc, std::map<int, std::vector<SymbolicScalar>>& addr2ValidShape,
                            std::map<int, std::vector<SymbolicScalar>>& addr2ValidShapeSpecified,
                            std::map<int, int>& magic2Coa);
    Status SetSubValidShape(Function& subFunc, std::map<int, std::vector<SymbolicScalar>>& addr2ValidShape,
                            std::map<int, std::vector<SymbolicScalar>>& addr2ValidShapeSpecified,
                            const std::map<int, int>& magic2Coa);
    Status UpdateParamIndex(Function& function);
    Status InferShape(Function& function);
};
} // namespace tile_fwk
} // namespace npu
#endif
