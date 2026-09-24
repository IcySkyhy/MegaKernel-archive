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
 * \file codegen_preproc.h
 * \brief
 */

#ifndef PASS_CODEGEN_PREPROC_H
#define PASS_CODEGEN_PREPROC_H

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/attribute.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu {
namespace tile_fwk {
class CodegenPreproc : public Pass {
public:
    CodegenPreproc() : Pass("CodegenPreproc") {}
    ~CodegenPreproc() override = default;
    Status RunOnFunction(Function& function) override;

private:
    /*
     * update oOperand of alloc node for codegen, gm tensor param layout:
     * [gm1 addr|gm1 offset0|gm1 offset1|...|gm1 tile shape0|gm1 tile shape1|...|gm1 raw shape0|gm1 raw shape1|...] ...
     * [gm1 addr|gm2 offset0|gm2 offset1|...|gm2 tile shape0|gm2 tile shape1|...|gm2 raw shape0|gm2 raw shape1|...] ...
     * so addr is always the first param of this gm tensor
     */
    Status SaveGmTensorParamIdxToOp(Function& func) const;
    // force combine axis
    Status ForceCombineAxisForAxisCombine(Function& func) const;
    bool IsCopyNeedSave(const Operation& op) const;
    void SetTensorParamAddr(LogicalTensor& tensor, int64_t tensorParamIdx, const SymbolicScalar& attrOffsetScalar,
                            const TensorAddrKey& key) const;
    void SetNeedAllocAttr(Function& function);
    void FixExpandDimForAxisCombine(Operation& op, int dimSize) const;
    std::string DumpOpList(Function& function);
    GmOutOfRangeCheckInfo ComputeGmOoRCheckInfo(const Operation& op, const std::vector<SymbolicScalar>& dynOffset,
                                                const std::vector<SymbolicScalar>& dynValidShape,
                                                const std::vector<SymbolicScalar>& rawShapeScalar,
                                                GmOutOfRangeCheckInfo::AccessType accessType) const;
    void GenGmOoRCheckInfoForOp(Operation& op) const;
    void GenGmOoRCheckInfo(Function& function) const;
    Status CheckSingleTensorAddrRange(const Operation& op, LogicalTensorPtr tensor, size_t tensorIdx,
                                      const std::string& tensorType) const;
    Status CheckTensorAddrRange(Function& function) const;
    bool combineAxis{false};
};

} // namespace tile_fwk
} // namespace npu
#endif // PASS_CODEGEN_PREPROC_H
